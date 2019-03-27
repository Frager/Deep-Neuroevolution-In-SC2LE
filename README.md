# Deep-Neuroevolution-In-SC2LE

Both manager and workers need to run for the Genetic Algorithm to work.
They don't have to be on the same machine, but they will need to be able to connect to the same 
broker/backend specified in celery_app.py

To run the manager (main GA code) in default configuration run "distributed_main.py"
if you want to change the default configurations, run with flags (see distributed_main.py)
example: 
"python distributed_main.py --map FindAndDefeatRoaches --use_minimap"

To run the workers (for model evaluations) run:
"celery -A evolution.tasks worker --loglevel=info --concurrency=12 -Ofair"
concurrency defines the number of workers (if not set: equal to CPU cores)

Saving and loading experiments:

Set --save_to 'experiment_name' when running distributed_main.py to specify save location (within experiments folder)
By default, save_to flag is "unnamed_experiment"

To continue an experiment (saved within experiments folder) run distributed_main.py with following tags:
--load_from experiment_name_to_load 
--save_to experiment_name (can be the same as experiment_name_to_load)
--gen generation_number_to_continue
Example: 
"python distributed_main.py --load_from experiment_name --save_to experiment_name --gen 50"

It can be usefull to run "python purge_tasks.py" to clean up the broker from leftover tasks after finishing an experiment
