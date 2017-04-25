# How to use

Create a batch feeder.

```
X_train, Y_train, X_test, Y_test = data_mnist()
train_data = make_batch_feeder({'x': X_train, 'y':Y_train})
test_data = make_batch_feeder({'x': X_test, 'y':Y_test})
```
Make list of add-ons.
```
    addons = [GlobalStep(),
                TrackAverages(), #do this before train
                Train(lambda gs: tf.train.AdadeltaOptimizer(learning_rate=learning_rate, #0.1
                                                            rho=0.95,
                                                            epsilon=1e-08), batch_size, train_feed={'epsilon' : epsilon}, loss = 'combined_loss', print_steps=print_steps),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = save_steps, checkpoint_path = os.path.join(train_dir, 'model.ckpt')),
                SummaryWriter(summary_steps = summary_steps, feed_dict = {}),
                Logger(),
                Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test (real)")]
```
Define the model.
```
def f():
	x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
	y = tf.placeholder(tf.float32, shape=(None, 10))
	# neural net here
	# give dictionary of relevant quantities
	model = {'loss': loss, 'inference': inference, 'accuracy': acc} # include any others you want to be able to access
	# give dictionary of placeholders
	ph_dict = {'x': x, 'y': y, 'epsilon': epsilon}
	return model, ph_dict
```
Make the trainer.
```
model, ph_dict = f()
trainer = Trainer(model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity, sess=sess)
```
Train.
```
trainer.init_and_train()
trainer.finish()
```
Example flags.
```
flags.DEFINE_string('train_dir', 'train_mix2/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mix.ckpt', 'Filename to save model under.')
flags.DEFINE_string('adv', 'fgsm', 'Type of adversary.')
#flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches. Must divide evenly into the dataset sizes. (FIX this later)')
#for batch_size = 100, is 600 times nb_epochs
flags.DEFINE_integer('max_steps', 3600, 'Number of steps to run trainer.')
flags.DEFINE_integer('print_steps', 100, 'Print progress every...')
flags.DEFINE_integer('eval_steps', 600, 'Run evaluation every...')
flags.DEFINE_integer('save_steps', 1200, 'Run evaluation every...')
flags.DEFINE_integer('summary_steps', 1200, 'Run summary every...')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('verbosity', 1, 'How chatty')
flags.DEFINE_string('load', 'T', 'Whether to load nets')
flags.DEFINE_string('load_from', 'pretrain/nets', 'Whether to run control experiment')
flags.DEFINE_integer('t', 100, 'Number of nets')
flags.DEFINE_integer('label_smooth', 0.1, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('reg_weight', 1, 'Weight on entropy regularizer')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
tf.app.flags.DEFINE_string('fake_data', False, 'Use fake data.  ')
```
