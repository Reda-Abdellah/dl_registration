from src.train import main

# # Simple call with defaults
results = main()

# With custom parameters
# results = main(
#     config_path='config.yaml',
#     experiment_name='my_first_experiment',
#     )

# # Resume from checkpoint programmatically
# results = main(
#     config_path='config.yaml',
#     resume_from='experiments/my_experiment/checkpoints/epoch_50.pth'
# )


# Access results
print(f"Best loss: {results['best_loss']}")
print(f"Experiment dir: {results['run_dir']}")
print(f"Best checkpoint: {results['best_checkpoint']}")
