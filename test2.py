from test_hawkes import PickledHawkesModel
pickle_filepath = 'hawkes_graph_relu_run_analysis.pickle'

# 2. Load and Analyze from Pickle using PickledHawkesModel
loaded_hawkes_model = PickledHawkesModel(pickle_filepath)

nll_results = loaded_hawkes_model.calculate_nll(nll_nonlinearities=['relu', 'linear', 'exp', 'power'])
print(f"\nNLL Results from PickledHawkesModel: {nll_results}")


