from Util.pickle_utils import load_file
from visualizations.validation_err_visualization import save_plot_accuracy


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


batch_size = [
    {
        'path': 'results/batch_size/validation/validation_accr_batch=50000_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 50000',
        'color': 'b',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=2048_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 2048',
        'color': 'k',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=1024_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 1024',
        'color': 'r',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=100_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 100',
        'color': 'g',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 32',
        'color': 'm',
    },
    {
        'path': 'results/batch_size/validation/validation_accr_batch=1_epochs=200_lr=0.01.pkl',
        'name': 'Paczka - 1',
        'color': 'y',
    },
]

activations = [
    {
        'path': './results/Activation/relu/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'ReLu',
        'color': 'b',
    },
    {
        'path': './results/Activation/sigmoid/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Sigmoid',
        'color': 'r'
    }
]

layers_plots = [
    {
        'path': 'results/Layer/one_layer_1_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '1 neuron',
        'color': 'b',
    },
    {
        'path': 'results/Layer/one_layer_10_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '10 neuronów',
        'color': 'r',
    },
    {
        'path': 'results/Layer/one_layer_28_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '28 neuronów',
        'color': 'c',
    },
    {
        'path': 'results/Layer/one_layer_50_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '50 neuronów',
        'color': 'm',
    },
    {
        'path': 'results/Layer/one_layer_300_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '300 neuronów',
        'color': 'y',
    },
    {
        'path': 'results/Layer/two_layers_100_10_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '100 i 10 neuronów',
        'color': 'k',
    },
    {
        'path': 'results/Layer/two_layers_100_50_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': '100 i 50 neuronów',
        'color': 'g',
    },
]

twolayers = [
    {
        'path': 'results/Layer/one_layer_5_neurons/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Ciąg walidacyjny',
        'color': 'g',
    },
    {
        'path': 'results/Layer/one_layer_5_neurons/training/training_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Ciąg treningowy',
        'color': 'r',
    },
]

weights_plots = [
    {
        'path': 'results/weigh_initializer/zero-initializer/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Zero',
        'color': 'b',
    },
    {
        'path': 'results/weigh_initializer/xavier-gain=6/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Xavier - gain 6',
        'color': 'r',
    },
    {
        'path': 'results/weigh_initializer/xavier-gain=1/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Xavier - gain 1',
        'color': 'g',
    },
    {
        'path': 'results/weigh_initializer/range-low=-0.05-high=0.05/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Uniform - range(-0.05, 0.05)',
        'color': 'm',
    },
    {
        'path': 'results/weigh_initializer/range-low=-1-high=1/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Uniform - range(-1, 1)',
        'color': 'k',
    },
    {
        'path': 'results/weigh_initializer/range-low=-2-high=2/validation/validation_accr_batch=32_epochs=200_lr=0.01.pkl',
        'name': 'Uniform - range(-2, 2)',
        'color': 'y',
    },
]

optimizers_plots = [
    {
        'path': './lab_3/Optimizer/AdaGrad Optimizer/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'AdaGrad',
        'color': 'r'
    },
    {
        'path': './lab_3/Optimizer/AdaDelta Optimizer/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'AdaDelta',
        'color': 'g'
    },
    {
        'path': './lab_3/Optimizer/Static Gradient Descent/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Static Gradient Descent',
        'color': 'b'
    },
    {
        'path': './lab_3/Optimizer/Momentum Optimizer/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Momentum',
        'color': 'y'
    },
    {
        'path': './lab_3/Optimizer/Adam Optimizer/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Adam Optimizer',
        'color': 'k'
    }
]

initializers = [
    {
        'path': './lab_3/Init/xavier-gain=6/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Xavier Initializer',
        'color': 'k'
    },
    {
        'path': './lab_3/Init/he-initializer-/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'He Initializer',
        'color': 'g'
    },
    {
        'path': './lab_3/Init/normal-distribution-loc=0-scale=1-a=10/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Normal Initializer',
        'color': 'b'
    }
]

costs = [
    {
        'path': './lab_3/cost/func=MSE&last_layer=softmax-stable/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Softmax - MSE',
        'color': 'g'
    },
    {
        'path': './lab_3/cost/func=CrossEntropy&last_layer=sigmoid/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Sigmoid - CrossEntropy',
        'color': 'b'
    },
    {
        'path': './lab_3/cost/func=MSE&last_layer=sigmoid/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Sigmoid - MSE',
        'color': 'y'
    },
    {
        'path': './lab_3/cost/func=CrossEntropy&last_layer=softmax-stable/validation/validation_accr_batch=32_epochs=30_lr=0.pkl',
        'name': 'Softmax - CrossEntropy',
        'color': 'k'
    }
]

# load(twolayers, 'Porównanie predykcji na zbiorze walidacyjnym i treningowym')
load(optimizers_plots, 'Wpływ optymalizatora wag na szybkość uczenia', 'optimizer_plots')
load(initializers, 'Wpływ sposobu inicjacji wag wag na szybkość uczenia', 'initializer_plots')
load(costs, 'Wpływ funkcji starty na szybkość uczenia', 'cost-function_plots')
