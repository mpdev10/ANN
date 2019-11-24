from Util.pickle_utils import load_file
from Util.visualizations.validation_err_visualization import save_plot_accuracy


def load(test_cases, title, time_title):
    data = []
    data_names = []
    data_times = []
    for test_case in test_cases:
        pickle_data, time = load_file(test_case['path'])
        x, y = zip(*pickle_data)
        data.append((x, y, test_case['name'], test_case['color']))
        data_names.append(test_case['name'])
        data_times.append(time)

    save_plot_accuracy(data, title, f'./plots/{time_title}.png')


inits = [
    {
        'path': './tests/initializers/xavier-gain=6/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Xavier Initializer',
        'color': 'k'
    },
    {
        'path': './tests/initializers/normal-distribution-loc=0-scale=1-a=10/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Normal Initializer',
        'color': 'b'
    }
]
if __name__ == '__main__':
    load(inits, 'Wpływ inicjalizacji wag na szybkość uczenia', 'initializer_plots')
