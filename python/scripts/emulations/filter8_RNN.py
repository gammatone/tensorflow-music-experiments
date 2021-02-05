#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-09 10:42:28
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
filter8_RNN.py
Try to emulate Joranalogue's Filter8 eurorack module using RNN model with Keras API.
The aim is to reproduce the cutoff and resonance effects of the filter applied on any input sound signal.
To do so the model is trained with some recorded data and labels described above:
    - Training dataset (temporal signals recorded for 25s at 44100Hz):
        * Input = white analog noise from Zlob Modular's Entropy module
        * Groundtruth = Input filtered by filter8
        * Labels = Cutoff + resonance params modulated by 2 asynchronous triangle LFOs (0.22Hz and 1.04Hz) from VCV rack
    - Testing dataset (temporal signals recorded for 17s at 44100Hz):
        * Input = Sawtooth wave from After Later Audio's Knit (uPlaits) at 260Hz (C4)
        * Groundtruth = Idem
        * Labels = Idem
The use of RNN type is motivated by its temporal behaviour which enables to make a real-time implementation later.
"""

import os, sys

import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model

import time

file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# Append python modules directory into python path
sys.path.append(os.path.join(root_dir, "python/modules/"))

# Custom imports
from dataset_utils import create_pkl_audio_dataset, get_dict_from_pkl, stack_array_from_dict_lastaxis
from dataset_utils import induce_IO_delay
from plot_utils import plot_by_key
from NN_utils import optimizer_call, vanilla_LSTM
from NN_utils import custom_loss
from dsp_utils import librosa_write_wav
from training_utils import KerasBufferizedNNHandler


# Comment this line if you have CUDA installed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


# tf.debugging.set_log_device_placement(True)


def get_RNN_model(in_length, out_length, n_features, stateful=False, batch_size=None):
    model = vanilla_LSTM(   in_length, out_length, n_features=n_features,
                            stateful=stateful,
                            batch_size=batch_size,
                            )
    return model

def training_routine(pkl_dir, input_signal_keys, output_signal_keys,
                    optimizer=optimizer_call(lr=1e-3),
                    loss_metric=custom_loss,
                    audio_out_dir=None, model_root_save_dir=None,
                    pkl_filenames=("train.pkl", "validation.pkl", "evaluation.pkl")
                    ):
    """
    Training function:
        - Load train & validation dataset
        - Create many-to-many RNN (e.g. IO_size=2048; hop_size=512)
        - Train RNN in stateless mode (i.e. truncated backprop for each chunks of length IO_size)
        - Evaluate trained model
        - Create audio file corresponding to the infered data from evaluation dataset
        - Save model and results (in .h5 and .pkl files)
    """
    # Define model handler for training
    my_model_trainer = KerasBufferizedNNHandler(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer, loss_metric,
                                                IO_SEQ_LENGTH, HOP_SIZE, IO_SEQ_LENGTH,
                                                need_training=True,
                                                train_filename=pkl_filenames[0],
                                                val_filename=pkl_filenames[1],
                                                eval_filename=pkl_filenames[2],
                                                )

    if NEED_PLOT:
        plot_by_key(my_model_trainer.train_dict, my_model_trainer.train_dict.keys(), title="Extract from train dataset",
                    start_idx=int(5e4), end_idx=int(5.1e4))
        plot_by_key(my_model_trainer.val_dict, my_model_trainer.val_dict.keys(), title="Extract from validation dataset",
                    start_idx=int(2e4), end_idx=int(2.1e4))
    my_model_trainer.prepare_datasets()
    # Define STATELESS RNN model
    model = get_RNN_model(in_length=IO_SEQ_LENGTH, out_length=IO_SEQ_LENGTH, n_features=N_FEATURES,
                            stateful=False,
                            )
    model.summary()

    ## MODEL TRAINING ##
    training_state = my_model_trainer.train_model(model, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                    val_freq=VALIDATION_FREQ,
                                                    need_sample_weight=True,
                                                    )
    # Define a dict for training losses
    training_losses = {}
    training_losses["loss"] = np.array(training_state.history["loss"])
    training_losses["val_loss"] = np.repeat(np.array(training_state.history["val_loss"]), VALIDATION_FREQ)
    plot_by_key(training_losses, 
            ["loss", "val_loss",],
            title="Losses value during training",
            )

    ## MODEL EVALUATION ##
    # Evaluate trained model (i.e. many-to-many samples model)  
    print("Evaluating trained model")
    eval_score = my_model_trainer.evaluate_model(model, need_plot=NEED_PLOT)
    print("Evaluation score = {}".format(eval_score))

    print("Inference on trained model")
    eval_predictions = my_model_trainer.predict(model, my_model_trainer.x_eval)
    print("Saving inference output to audio file")

    ## AUDIO FILE WRITING ##
    if audio_out_dir is not None:
        # Save to audio file
        if not os.path.exists(audio_out_dir):
            os.makedirs(audio_out_dir)
        savefilepath = os.path.join(audio_out_dir, "infered_data_trained_model.wav")
        # Use numpy asfortranarray() function to ensure Fortran contiguity on the array
        librosa_write_wav(np.asfortranarray(eval_predictions), savefilepath, sr=SAMPLE_RATE)

    ## MODEL SAVING ##
    model_save_dir = None
    if model_root_save_dir is not None:
        # Save model and results into a sub directory specified by current data/time
        import datetime
        model_save_dir = os.path.join(model_root_save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(model_save_dir)
        # Save useful constants from training as pickle
        model_infos_dict = {}
        model_infos_dict["sample_rate"] = SAMPLE_RATE
        model_infos_dict["epochs"] = EPOCHS
        model_infos_dict["batch_size"] = BATCH_SIZE
        model_infos_dict["hop_size"] = HOP_SIZE
        model_infos_dict["loss_metric"] = loss_metric
        model_infos_dict["optimizer"] = optimizer
        model_infos_dict["val_freq"] = VALIDATION_FREQ
        model_infos_dict["training_losses"] = training_losses
        model_infos_dict["score_many2many"] = eval_score
        with open(os.path.join(model_save_dir,"model_infos.pkl"), "wb") as f:
            pickle.dump(model_infos_dict, f)
        # Save model in the sub directory
        model.save(os.path.join(model_save_dir,'trained_model'), save_format='tf')

    return model_save_dir

def testing_routine(savedmodel_dir, pkl_dir, input_signal_keys, output_signal_keys,
                    audio_out_dir=None, test_pkl_filename="evaluation.pkl",
                    ):
    """
    Testing function:
        - Load a pretrained many-to-many RNN Keras model
        - Create one-to-one stateful RNN (i.e. the real-time implementation model)
        with same graph than pretrained model
        - Copy weight from the pretrained model
        - Evaluate one-to-one model
        - Create audio file corresponding to the infered data from a test dataset
    """
    ## PRETRAINED MODEL LOADING ##
    # Load pretrained stateless model
    for file in os.listdir(savedmodel_dir):
        if file == "trained_model":
            pretrained_model = load_model(os.path.join(savedmodel_dir, file), custom_objects={'custom_loss': custom_loss})
            break
    # Load training infos from saved dictionary
    with open(os.path.join(savedmodel_dir, "model_infos.pkl"), "rb") as f:
        model_infos_dict = pickle.load(f)
    # Deduce optimizer and loss metric to be able to compile real-time model later
    optimizer = model_infos_dict["optimizer"]
    loss_metric = model_infos_dict["loss_metric"]
    # Plot losses from training
    plot_by_key(model_infos_dict["training_losses"], 
            ["loss", "val_loss",],
            title="Losses value during training",
            xaxis_str="Epochs",
            yaxis_str="ESR"
            )

    ## REAL-TIME MODEL CREATION ##
    # Define model handler for testing one-to-one stateful model:
    # means only evaluation dataset (no train and validation datasets)
    # means io_seq_length=1; hop_size=1; 
    my_rt_model_tester = KerasBufferizedNNHandler(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer, loss_metric,
                                                1, 1, 1,
                                                need_training=False,
                                                eval_filename=test_pkl_filename,
                                                )
    my_rt_model_tester.prepare_datasets()

    print("\nOne-to-one RNN model (i.e. the one used in real-time) creation according to trained model config")
    rt_model = get_RNN_model(in_length=1, out_length=1, n_features=N_FEATURES,
                            stateful=True,
                            batch_size=1,
                            )
    rt_model.summary()
    # Copy weights from pretrained model
    training_weights = pretrained_model.get_weights()
    rt_model.set_weights(training_weights)

    ## REAL-TIME MODEL EVALUATION ##
    print("Evaluating real-time model")
    EVAL_DURATION_s = 0.1
    eval_score = my_rt_model_tester.evaluate_model(rt_model, batch_size=1,
                                                    start_end_idxs=(0, int(EVAL_DURATION_s * SAMPLE_RATE)),
                                                    need_plot=NEED_PLOT)
    print("Evaluation score = {}".format(eval_score))
    print("Inference on real-time model")
    # Predict output of evaluation dataset inputs from stateful model
    eval_predictions = my_rt_model_tester.predict(rt_model, my_rt_model_tester.x_eval, batch_size=1)
    print("Saving inference output to audio file")

    ## AUDIO FILE WRITING ##
    # Save to audio file
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir)
    savefilepath = os.path.join(audio_out_dir, "infered_data_realtime_model.wav")
    # Use numpy asfortranarray() function to ensure Fortran contiguity on the array
    librosa_write_wav(np.asfortranarray(eval_predictions), savefilepath, sr=SAMPLE_RATE)

    ## MODEL SAVING ##
    # Save useful constants from training as pickle
    model_infos_dict["score_one2one"] = eval_score
    with open(os.path.join(savedmodel_dir,"model_infos.pkl"), "wb") as f:
        pickle.dump(model_infos_dict, f)
    run_model = tf.function(lambda x: rt_model(x))
    concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 1, 3], rt_model.inputs[0].dtype))
    rt_model.save(os.path.join(savedmodel_dir,'trained_model_rt'), save_format='tf', signatures=concrete_func)

    # save model in .h5 format as well
    rt_model.save(os.path.join(savedmodel_dir,'trained_model_rt.h5'), signatures=concrete_func)

    return

def tflite_routine(savedmodel_dir, pkl_dir, input_signal_keys, output_signal_keys,
                    test_pkl_filename="evaluation.pkl",
                    ):
    """
    TensorFlow Lite conversion function:
        - Load a pretrained one-to-one RNN Keras model
        - Convert it to tensorflow lite format
        - Evaluate tflite model
    """
    ## PRETRAINED MODEL LOADING ##
    # Load pretrained one2one stateful model
    for file in os.listdir(savedmodel_dir):
        if file == "trained_model_rt":
            stateful_model = load_model(os.path.join(savedmodel_dir, file), custom_objects={'custom_loss': custom_loss})
            stateful_model.reset_states()
            break
    # Load training infos from saved dictionary
    with open(os.path.join(savedmodel_dir, "model_infos.pkl"), "rb") as f:
        model_infos_dict = pickle.load(f)

    # If RNN model to convert is stateful. Conversion won't work...
    # So create a stateless model with same parameters and convert it to tensorflow lite
    # The tflite model is stateful by default so it should work as the keras stateful model
    print("\nOne-to-one RNN model (i.e. the one used in real-time) creation according to trained model config")
    stateless_model = get_RNN_model(in_length=1, out_length=1, n_features=N_FEATURES,
                            stateful=False,
                            batch_size=1,
                            )
    # Copy weights from pretrained model
    training_weights = stateful_model.get_weights()
    stateless_model.set_weights(training_weights)

    ## TFLITE CONVERSION ##
    target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(stateless_model)
    # tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_converter.target_ops = target_ops
    tflite_converter.target_spec.supported_ops = set(target_ops)
    # Call the representative dataset generator
    # tflite_converter.representative_dataset = self.repr_dataset_gen
    # Run the TFLite converter
    tflite_model = tflite_converter.convert()

    ## TFLITE EVALUATION ##
    # open evaluation set from pickle file
    eval_dict = get_dict_from_pkl(os.path.join(pkl_dir, test_pkl_filename))
    x_eval = stack_array_from_dict_lastaxis(eval_dict, input_signal_keys)
    y_eval = stack_array_from_dict_lastaxis(eval_dict, output_signal_keys)
    print("Evaluating tflite conversion")
    # Prepare TensorFlow Lite Interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Allocate output arrays, each for the 2 models to compare
    stateful_outputs = np.zeros((int(EVAL_DURATION_s * SAMPLE_RATE),), dtype=np.float32)
    tflite_outputs = np.zeros(stateful_outputs.shape, dtype=np.float32)
    # Allocate dt array to store inference times
    dt_array = np.zeros(stateful_outputs.shape, dtype=np.float32)
    # Inference loop
    for i in range(int(EVAL_DURATION_s * SAMPLE_RATE)):
        x_input = np.expand_dims(x_eval[i:i+1, :], axis=0)
        stateful_outputs[i] = stateful_model.predict(x_input)[0][0]
        # Inference on tflite model
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], x_input)
        interpreter.invoke()
        tflite_outputs[i] = interpreter.get_tensor(output_details[0]["index"])[0][0]
        end_time = time.perf_counter()
        dt_array[i] = end_time - start_time


    print("'tflite vs. keras stateful model' MSE = {}".format(((stateful_outputs - tflite_outputs)**2).mean(axis=0)))
    print("At current samplerate ({} Hz), rt sample-wise inference time should last less than {} s".format(SAMPLE_RATE, 1/SAMPLE_RATE))
    print("Evaluated tflite sample-wise inference time (s) = (mean: {}; std: {})".format(dt_array.mean(axis=0), dt_array.std(axis=0)))

    ## TFLITE SAVING ##
    # Save quantized model
    with open(os.path.join(savedmodel_dir,"trained_model_rt.tflite"), "wb") as f:
        f.write(tflite_model)

def main():
    """
    Main routine
    """
    global NEED_PLOT
    global SAMPLE_RATE
    global SAMPLE_DELAY
    global IO_SEQ_LENGTH, HOP_SIZE, N_FEATURES
    global EPOCHS, BATCH_SIZE, VALIDATION_FREQ
    global EVAL_DURATION_s
    # define plot flags
    NEED_PLOT = True
    # define training, testing, tflite flags
    NEED_TRAIN = True
    NEED_TEST = True
    NEED_TFLITE = True

    # define audio parameters
    SAMPLE_RATE = 44100
    # Delay induced to the filter output
    SAMPLE_DELAY = 0

    # Training hyper parameters
    IO_SEQ_LENGTH = 4096
    HOP_SIZE = 128
    N_FEATURES = 3
    BATCH_SIZE = 32
    EPOCHS = 1000
    VALIDATION_FREQ = 5

    # Evaluation params
    EVAL_DURATION_s = 0.01

    # Define signal keys
    input_signal_keys = ["signal_in", "cutoff", "resonance"]
    output_signal_keys = ["signal_filtered"]
    signal_keys = input_signal_keys + output_signal_keys

    # define pickle directories (where the pickle files are located)
    pkl_dir = os.path.join(root_dir, "data/pickle/filter8_rec")
    pkl_train_filename = "train.pkl"
    pkl_val_filename = "validation.pkl"
    pkl_eval_filename = "evaluation.pkl"
    my_pkl_filenames = (pkl_train_filename, pkl_val_filename, pkl_eval_filename)
    # define audio directories (where the audio files are located)
    audio_dir = os.path.join(root_dir, "data/audio/filter8_rec")
    audio_train_dir = os.path.join(audio_dir, "white_noise")
    audio_val_dir = os.path.join(audio_dir, "saw_wave")
    audio_eval_dir = os.path.join(audio_dir, "saw_wave")
    # Define an audio output directory
    audio_out_dir = os.path.join(audio_dir, "inference")
    my_audio_dirs = (audio_train_dir, audio_val_dir, audio_eval_dir)
    # Define where to save trained model
    model_root_save_dir = os.path.join(root_dir, "data/keras_models/RNN")

    # Save to pickle if dataset does not exist yet
    # First make sure pickle directories exist
    for i in range(len(my_pkl_filenames)):
        # Create pickle dictionary
        create_pkl_audio_dataset(my_audio_dirs[i], pkl_dir, my_pkl_filenames[i], keynames=signal_keys, sr=SAMPLE_RATE)
        # Induce IO delay if needed
        induce_IO_delay(os.path.join(pkl_dir, my_pkl_filenames[i]),
                        input_signal_keys,
                        output_signal_keys,
                        num_samples=SAMPLE_DELAY,
                        )
    savedmodel_dir = None

    if NEED_TRAIN:
        # Train, evaluate and save many2many RNN model
        savedmodel_dir = training_routine(pkl_dir, input_signal_keys, output_signal_keys,
                                        optimizer=optimizer_call(lr=1e-3),
                                        loss_metric=custom_loss,
                                        audio_out_dir=audio_out_dir, model_root_save_dir=model_root_save_dir,
                                        )

    # If no training step, specify manually path where to load model
    if savedmodel_dir is None:
        savedmodel_dir = os.path.join(model_root_save_dir, "2021-01-23_09-14-04")

    if NEED_TEST:
        # Load pretrained model, convert it to one2one (i.e. sample-wise real-time) RNN model, evaluate & save it
        testing_routine(savedmodel_dir, pkl_dir,
                        input_signal_keys, output_signal_keys,
                        audio_out_dir=audio_out_dir, test_pkl_filename="evaluation.pkl",
                        )

    if NEED_TFLITE:
        # Load one2one RNN model, convert to tflite format, evaluate & save it
        tflite_routine(savedmodel_dir, pkl_dir,
                        input_signal_keys, output_signal_keys,
                        test_pkl_filename="evaluation.pkl",
                        )


if __name__ == "__main__":
    main()
