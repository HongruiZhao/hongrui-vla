* I would like to modify @scripts/train.py:
    * Use `einops` for reshaping instead of `permute`
    * Always import `cv2` and get rid of if else statements that assume `cv2` is not imported.
    * Add tensorboard to track `avg_loss` at each episode.
    * Create a configuration file `training.yaml` in @configs to handle all the hyperparameters and get rid of most of the argparse.
    * Only keep one argparse argument for load the configuration file.