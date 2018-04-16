#!/usr/bin/python
# -*- coding: UTF-8 -*- 
import tensorflow as tf

import random
import numpy as np
import matplotlib.pyplot as plt


BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 20     # backpropagation through time 的 time_steps
BATCH_SIZE = 50     
INPUT_SIZE = 1      # sin 数据输入 size
OUTPUT_SIZE = 1     # cos 数据输出 size
CELL_SIZE = 10      # RNN 的 hidden unit size 
LR = 0.006          # learning rate


class LstmRnnModel(object):

    def __init__(self, nStep, inputSize, outputSize, cellSize, batchSize, layerName = "LSTM_RNN"):
        """
        LSTM RNN bp
            err       err    err                              err       err    err                 
             |    /    |  /   |                                |    /    |  /   |                 
initState  cell   -  cell - cell -- finnalState = initState  cell   -  cell - cell -- finnalState 
             |    /    |  /   |                                |    /    |  /   |                 
            in        in.    in                               in        in.    in                 
            ------- nStep -----
            -----------------------------  batchSize. --------------------------------------------
        nStep: 每次BP的步数
        inputSize 每隔in的输入大小
        outputSize 每次out的输出大小
        cellSize 单元大小
        batchSize 每次训练时，batch的大小
        """
        self.nStep = nStep
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.cellSize = cellSize
        self.batchSize = batchSize
        self.layerName = layerName
        self.weights = {}
        self.cellInitState = None
        self.cellFinalState = None
        self.cellOutput = None
        self.yPrediction = None
        self.loss = None
        self.train = None

    def build(self, x, y):
        """
        x: input placeholder
        y: output placeholder
        """
        with tf.name_scope(self.layerName):
            layer = self.add_input_layer(x)
            output, state = self.add_cell(layer)
            self.yPrediction = self.add_output_layer(output)
            self.loss = self.gen_loss(y, self.yPrediction)
            self.train = self.gen_train()
        return self.yPrediction

    def gen_train(self):
        with tf.name_scope("train"):
            train = tf.train.AdamOptimizer(LR).minimize(self.loss) 
        return train

    def gen_loss(self, y, yPrediction):
        with tf.name_scope("loss"):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(yPrediction, [-1], name='reshape_pred')]
                , [tf.reshape(y, [-1], name='reshape_target')]
                , [tf.ones([self.batchSize * self.nStep * self.outputSize], dtype=tf.float32)]
                , average_across_timesteps=True
                , softmax_loss_function = LstmRnnModel.ms_error
                , name='sequence_loss_by_example'
            )
            loss = tf.div(
                tf.reduce_sum(loss, name='losses_sum'),
                self.batchSize,
                name='average_loss'
            )
            tf.summary.scalar('loss', loss)
        return loss

    @staticmethod
    def ms_error(labels, logits):
        with tf.name_scope("ms_error"):
            return tf.square(tf.subtract(labels, logits))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def add_input_layer(self, x):
        with tf.name_scope("input_layer"):
            xInput = tf.reshape(x, [-1, self.inputSize], name="reshape_input")
            self.weights["W_in"] = self.weight_variable([self.inputSize, self.cellSize])
            self.weights["bias_in"] = self.bias_variable([self.cellSize])
            with tf.name_scope("Wx_plus_b"):
                layer = tf.matmul(xInput, self.weights["W_in"]) + self.weights["bias_in"]
                layer = tf.reshape(layer, [-1, self.nStep, self.cellSize], name = "reshape_for_cell")
        return layer

    def add_cell(self, layer):
        with tf.name_scope("rnn_cell"):
            lstmCell = tf.nn.rnn_cell.BasicLSTMCell(self.cellSize, forget_bias = 1.0, state_is_tuple = True)
            with tf.name_scope("inital_state"):
                self.cellInitState = lstmCell.zero_state(self.batchSize, dtype = tf.float32)
            self.cellOutput, self.cellFinalState = tf.nn.dynamic_rnn(
                lstmCell, layer, initial_state=self.cellInitState, time_major=False
            )
        return self.cellOutput, self.cellFinalState

    def add_output_layer(self, output):
        with tf.name_scope("output_layer"):
            # reshape to [batchSize * nStep, cellSize]
            output = tf.reshape(output, [-1, self.cellSize], name = "reshape_output")
            self.weights["W_out"] = self.weight_variable([self.cellSize, self.outputSize])
            self.weights["bias_out"] = self.bias_variable([self.outputSize])
            with tf.name_scope("Wx_plus_b"):
                output = tf.matmul(output, self.weights["W_out"]) + self.weights["bias_out"]
        return output


def get_batch():
    global BATCH_START, TIME_STEPS, BATCH_SIZE
    xData = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    ySeq = np.sin(xData)
    yRes = np.cos(xData) + 0.2
    BATCH_START += TIME_STEPS
    return [ySeq[:,:, np.newaxis], yRes[:,:,np.newaxis], xData]

x = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE], name = "x")    
y = tf.placeholder(tf.float32, [None, TIME_STEPS, OUTPUT_SIZE], name = "y")


if __name__ == '__main__':
    model = LstmRnnModel(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    model.build(x, y)
    plt.ion()   # 设置连续 plot
    plt.show()
    with tf.Session() as sess:
        # 可视化
        logDir = "logs/rnn_regression"
        # summaries合并
        merged = tf.summary.merge_all()
        trainWriter = tf.summary.FileWriter(logDir + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(200):
            ySeq, yRes, xData = get_batch()
            if i == 0:
                feedDict = {
                    x: ySeq
                    , y: yRes
                }
            else:
                feedDict = {
                    x: ySeq
                    , y: yRes
                    , model.cellInitState : lastFinalState # 保持 state 的连续性
                }
            _, loss, lastFinalState, yPrediction = sess.run(
                [model.train, model.loss, model.cellFinalState, model.yPrediction]
                , feed_dict = feedDict
                )
            if i % 20 == 0:
                print("loss: ", round(loss, 4))
                result = sess.run(merged, feed_dict = feedDict)
                trainWriter.add_summary(result, i)
            # draw image
            plt.plot(xData[0].flatten(), ySeq[0].flatten(), 'y-.', label = "trainSeq")
            plt.plot(xData[0].flatten(), yRes[0].flatten(), 'r', label = "real")
            plt.plot(xData[0].flatten(), yPrediction.reshape(BATCH_SIZE, TIME_STEPS)[0].flatten(), "b--", label = "prediction")
            plt.ylim((-1.4, 1.4))
            #plt.legend()
            plt.draw()
            plt.pause(0.3)
        trainWriter.close()


