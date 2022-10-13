# snowfall

## DEPRECATED: See [Icefall](https://github.com/k2-fsa/icefall) instead.

**This repo is deprecated in favor of its successor project [Icefall](https://github.com/k2-fsa/icefall).**

## About

Snowfall is an early draft of what will eventually be "icefall", the official recipes associated with k2 and lhotse. At
the moment it is some early drafts of recipes, that we'll use for debugging and collaboration while the overall shape of
the project becomes clearer.

## Diagnostics

Our diagnotics are automatically collected using TensorBoard. You can inspect it locally by running:

    $ tensorboard --logdir <exp_dir>

And then entering the url `localhost:6006` in your browser (it is possible to change the port with `--port` option).
When running the expts on a remote server, use port forwarding with SSH (`ssh -L 6006:localhost:6006 user@address`)
so that your browser can connect to tensorboard.

Some noteworthy tensorboard options:

- "Toggle all runs" to enable/disable all of the layers' plots
- "Ignore outliers in chart scaling" on/off is useful depending on the plot;
- "Tooltip sorting method" descending/ascending/closest, depending on the plot;
- I usually set smoothing to 0;
- I added a plot for "epoch" so it's easy to check at which step an epoch starts/ends.

Finally, Google added a service called "tensorboard.dev" where you can host your tensorboard data to show the exp to
others. To do that, use the command: `tensorboard dev upload --logdir <exp_dir>` and follow the instructions in the
terminal.
