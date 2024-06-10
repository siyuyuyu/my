import numpy as np


class BinhKornTest:
    """
    Class for BinhKorm problem
    """

    def binh_korn(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same size.")

        f1 = 4 * x**2 + 4 * y**2
        f2 = (x - 5) ** 2 + (y - 5) ** 2
        result = np.column_stack((f1, f2))
        return result

    def g1(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        return (x - 5) ** 2 + y**2 - 25

    def g2(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        return (x - 8) ** 2 + (y + 3) ** 2 - 7.7

    def generate_dataset(self, num_samples):
        """
        :param num_samples:
        :return:
        """
        x_samples = []
        y_samples = []
        num_generated = 0

        while num_generated < num_samples:
            x = np.random.uniform(low=0, high=5)
            y = np.random.uniform(low=0, high=3)
            g1_constraint = self.g1(x, y) <= 0
            g2_constraint = self.g2(x, y) <= 0

            if g1_constraint or g2_constraint:
                x_samples.append(x)
                y_samples.append(y)
                num_generated += 1

        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        f = self.binh_korn(x_samples, y_samples)

        return np.column_stack((x_samples, y_samples)), f


class ZDT1Problem:
    """
    DT1 Problem:
        Minimize f1(x) and f2(x) subject to the constraints:
            f1(x) = x[0]
            g(x) = 1 + 9/(n-1) * sum(x[1:])
            f2(x) = g(x) * (1 - sqrt(f1/g) - (f1/g) * sin(10*pi*f1))
            0 <= x[i] <= 1 for i = 0, 1, ..., n-1
    """

    def zdt1(self, x):
        """
        Calculates the objective function values for the ZDT1 problem.

        Parameters:
            x (numpy.ndarray): Array of decision variable values.
                The length of x determines the number of decision variables.

        Returns:
            numpy.ndarray: Array of objective function values [f1, f2].


        """
        n = len(x)

        f1 = x[0]

        g = 1 + 9 / (n - 1) * np.sum(x[1:])

        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))

        return np.array([f1, f2])

    def generate_dataset(self, num_samples, num_variables):
        """
        Generates a dataset for the ZDT1 problem.

        Parameters:
            num_samples (int): Number of samples in the dataset.
            num_variables (int): Number of decision variables.

        Returns:
            numpy.ndarray: Array of shape (num_samples, num_variables) containing the decision variable values.
            numpy.ndarray: Array of shape (num_samples, 2) containing the corresponding objective function values.



        Example

        problem = ZDT1Problem()
        num_samples = 100
        num_variables = 2
        X, objectives = problem.generate_dataset(num_samples, num_variables)

        """
        X = np.random.rand(num_samples, num_variables)
        objectives = np.array([self.zdt1(x) for x in X])

        return X, objectives
