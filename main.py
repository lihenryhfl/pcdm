# Heavily inspired by VDM codebase.

import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # Disable TF info/warnings # nopep8

import jax
import tensorflow as tf
from absl import logging
from absl import flags
from absl import app
from ml_collections import config_flags
from pcdm.utils import get_workdir
import pcdm.experiment_pcdm

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("checkpoint", '', "Checkpoint to evaluate.")
flags.DEFINE_string("mode", 'train', "train / eval")
flags.mark_flags_as_required(["config", "workdir"])
flags.DEFINE_string("log_level", "info", "info/warning/error")


def main(argv):
  del argv
  if jax.process_index() == 0:
    logging.set_verbosity(FLAGS.log_level)
  else:
    logging.set_verbosity("error")
  logging.warning("=== Start of main() ===")

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX. (Not necessary with TPU.)
  #tf.config.experimental.set_visible_devices([], "GPU")

  logging.info("JAX process: %d / %d",
               jax.process_index(), jax.process_count())
  #logging.info("JAX devices: %r", jax.devices())

  experiment = pcdm.experiment_pcdm.Experiment_PCDM(FLAGS.config)

  if FLAGS.mode == "train":
    workdir = os.path.join(FLAGS.workdir, get_workdir())
    logging.info("Training at workdir: "+FLAGS.workdir)
    experiment.train_and_evaluate(workdir)
  elif FLAGS.mode == "eval":
    experiment.evaluate(FLAGS.workdir, FLAGS.checkpoint)
  else:
    raise Exception("Unknown FLAGS.mode")

if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
