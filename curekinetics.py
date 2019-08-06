# Define the CureKinetics class
import numpy as np
import os
import xlrd

GAS_CONSTANT = 8.314e-3  # kJ/mol


class CureKinetics:

    def __init__(self, freq_factor, act_energy, reaction_order, autocatalytic_order=0,
                 heat_of_reaction=0.):
        """
        Initialize the cure kinetics model for a single reaction

        :param freq_factor: float Frequency factor (s^-1)
        :param act_energy: float Activation energy (kJ/mol)
        :param reaction_order: float n factor reaction order for nth order reaction
        :param autocatalytic_order: float m factor reaction order for autocatalytic model
        """
        self.freq_factor = freq_factor
        self.act_energy = act_energy
        self.reaction_order = reaction_order
        self.autocatalytic_order = autocatalytic_order
        self.heat_of_reaction = heat_of_reaction

    def compute_mechanism_function(self, conversion):
        """
        Return the mechanism function for the given level of conversion

        Verify that the degree of conversion is between 0 and 1 and is a floating point value

        :param conversion: float degree of conversion
        :return: float mechanism function value
        """
        try:
            conversion = float(conversion)
        except ValueError:
            raise Exception("Degree of conversion must be a float")
        if (conversion < 0.) or (conversion > 1.0):
            raise Exception("Degree of conversion must be in range [0, 1]", conversion, 'given')
        return conversion ** self.autocatalytic_order * (1.0 - conversion) ** self.reaction_order

    def compute_arrhenius(self, temperature):
        """
        Return the value of the compute_arrhenius equation for the given temperature

        :param temperature: float temperature in K
        :return: float compute_arrhenius value in (1/s)
        """
        return self.freq_factor * np.exp(-self.act_energy/(GAS_CONSTANT*temperature))

    def compute_rate(self, temperature, conversion):
        """
        Return the reaction rate for the model with the current temperature and degree of conversion

        :param temperature: float Temperature in K
        :param conversion: float degree of conversion
        :return: float conversion rate in (1/s)
        """
        return self.compute_arrhenius(temperature) * self.compute_mechanism_function(conversion)

    def update_conversion(self, temperature, conversion, delta_time, return_heat_flow=False):
        """
        Return the new conversion with a linear change over the given time step
        
        :param temperature: float Temperature in K
        :param conversion: float degree of conversion at beginning of time step
        :param delta_time: float length of time step in s
        :param return_heat_flow: bool set to true for returning heat flow as well as conversion
        :return: float conversion after delta_time
                 optional float heat flow for the current step
        """
        reaction_rate = self.compute_rate(temperature, conversion)
        updated_conversion = conversion + delta_time * reaction_rate

        if return_heat_flow:
            heat_flow = self.heat_of_reaction * reaction_rate
            return min(updated_conversion, 1.0), heat_flow
        else:
            return min(updated_conversion, 1.0)

    def compute_conversion_history(self, time_array, temperature_array, return_heat_flow=False):
        """
        Return the conversion history for the current model for the given temperature/time history

        :param time_array: n-length np.ndarray of floats for the time points for analysis
        :param temperature_array: n-length np.ndarray of floats for the time points for analysis
        :param return_heat_flow: bool to return the heat flow history as well (default=False)

        :return: np.ndarray (n, ) floats of conversion history at given time points
            optional np.ndarray (n, ) floats of heat flow history at given time points
        """
        n = time_array.shape[0]
        m = temperature_array.shape[0]
        if m != n:
            raise Exception('Length of the temperature array(' + str(m) +
                            ') must be equal to the length of the time array (' + str(n) + ')')
        conversion_history = np.zeros(n)
        heat_flow = np.zeros(n)

        # Initialize a small, non-zero conversion
        conversion_history[0] = 1e-10

        for i in range(1, n):
            delta_time = time_array[i] - time_array[i - 1]
            conversion_history[i], heat_flow[i] = self.update_conversion(temperature_array[i-1],
                                                                         conversion_history[i-1],
                                                                         delta_time,
                                                                         return_heat_flow=True)
        if return_heat_flow:
            return conversion_history, heat_flow
        else:
            return conversion_history

    def compute_isothermal_history(self, temperature, dt=1.0):
        """
        Return the conversion history, time history, and heat flow history for the given isothermal case

        :param temperature: float temperature for isothermal condition
        :param dt: float time step for explicit solution (default=1.0)
        :return: np.ndarray conversion values at time points
                 np.ndarray time points
                 np.ndarray heat flow values at time points
        """
        i = 0
        degree_conversion = 1e-10
        conversion_history = [degree_conversion, ]
        time_history = [0., ]
        heat_flow_history = [0., ]

        while degree_conversion < 1.0 - 1e-4:
            i += 1
            degree_conversion, heat_flow = self.update_conversion(temperature, degree_conversion,
                                                                  dt, return_heat_flow=True)
            time_history.append(time_history[i-1] + dt)
            conversion_history.append(degree_conversion)
            heat_flow_history.append(heat_flow)

            if i == 100000 and degree_conversion < 0.5:
                raise Warning('After 100000 conversion is only {0:9.5e}, consider changing dt or temperature'.format(degree_conversion))

        return conversion_history, time_history, heat_flow_history

    def compute_ramp_history(self, ramp_rate, dt=1.0):
        """
        Return the conversion history, time history, and heat flow history for the given non-isothermal case

        Begin the ramp from -100C and continue through conversion

        :param ramp_rate: float temperature ramp rate in C/min
        :param dt: float time step for explicit solution (default=1.0)
        :return: np.ndarray conversion values at time points
                 np.ndarray time points
                 np.ndarray heat flow values at time points
        """
        i = 0
        degree_conversion = 1e-10
        conversion_history = [degree_conversion, ]
        time_history = [0., ]
        temperature_history = [173.15, ]
        heat_flow_history = [0., ]

        while degree_conversion < 1.0 - 1e-4:
            i += 1
            degree_conversion, heat_flow = self.update_conversion(temperature_history[i-1], degree_conversion,
                                                                  dt, return_heat_flow=True)
            temperature_history.append(temperature_history[i-1] + ramp_rate * dt / 60.)
            time_history.append(time_history[i-1] + dt)
            conversion_history.append(degree_conversion)
            heat_flow_history.append(heat_flow)

            if i == 100000 and degree_conversion < 0.5:
                raise Warning('After 100000 conversion is only {0:9.5e}, consider changing dt or temperature'.format(degree_conversion))

        return conversion_history, time_history, temperature_history, heat_flow_history


class DSCData:

    def __init__(self, file_path):
        """
        Read in the dsc data file

        :param file_path: string with the absolute path to the data file
        """
        if not os.path.exists(file_path):
            raise Exception('Could not find file at', file_path)

        self._file_path = file_path
        self.data = []
        self.temperature = np.zeros(1)
        self.time = np.zeros(1)
        self.heat_flow = np.zeros(1)
        self.sample_mass = 0.
        self.peak_heat_flow = 0.
        self.peak_heat_flow_temp = 0.
        self.heat_of_reaction = 0.
        self.data_bounds = [0, 0]
        self.read_file()
        self.compute_heat_flow()

    def read_file(self):
        """
        Read in the data file and store the data in the class

        :return: None
        """
        book = xlrd.open_workbook(self._file_path)
        sh = book.sheet_by_index(0)

        sample_mass = 1.
        sample_mass_found = False
        in_data = False
        data = []
        for n_row in range(sh.nrows):
            row = sh.row(n_row)
            if in_data:
                data.append([row[i].value for i in range(4)])
            elif 'StartOfData' in row[0].value:
                in_data = True
            # Find the data line with the sample mass
            elif 'Size' in row[0].value:
                sample_mass_found = True
                self.sample_mass = row[1].value

        data = np.array(data)
        self.data = data

        time = data[:, 0] * 60
        temperature = data[:, 1]

        if sample_mass_found:
            heat_flow = data[:, 2] / self.sample_mass
        else:
            heat_flow = data[:, 2]
            # print('Sample mass could not be found')

        self.temperature = np.array(temperature)
        self.time = np.array(time)
        self.heat_flow = np.array(heat_flow)

    def compute_heat_flow(self):
        # Find the peak of the heat flow
        self.peak_heat_flow = self.heat_flow.max()
        peak_heat_flow_idx = self.heat_flow.argmax()
        self.peak_heat_flow_temp = self.temperature[peak_heat_flow_idx]
        # print('Peak heat flow:', peak_heat_flow, 'mW/mg at temperature: ', peak_heat_flow_temp, 'C')

        # Find the point where the heat flow goes to 0 after the peak
        sample_max_idx = len(self.time)
        for i in range(peak_heat_flow_idx, sample_max_idx):
            if self.heat_flow[i] <= 0.:
                sample_max_idx = i
                break

        sample_min_idx = 0
        for i in range(peak_heat_flow_idx, 0, -1):
            if self.heat_flow[i] <= 0.:
                sample_min_idx = i
                break

        # Limit the heat flow curve to the positive section
        time = self.time[sample_min_idx: sample_max_idx] - self.time[sample_min_idx]
        heat_flow = self.heat_flow[sample_min_idx: sample_max_idx]

        # Compute the heat of reaction
        heat_of_reaction = np.trapz(heat_flow, time)
        # print('Heat of reaction:', heat_of_reaction, 'J/g')
        self.heat_of_reaction = heat_of_reaction
        self.data_bounds = [sample_min_idx, sample_max_idx]

