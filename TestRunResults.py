class TestRunResults:
    def __init__(self, averages, std_deviations, test_average, test_std_deviation, train_averages):
        self.averages = averages
        self.std_deviations = std_deviations
        self.test_average = test_average
        self.test_std_deviation = test_std_deviation
        self.train_averages = train_averages
