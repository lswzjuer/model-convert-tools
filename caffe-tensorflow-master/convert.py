#!/usr/bin/env python
#coding:utf-8
import numpy as np
import argparse
from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer


def fatal_error(msg):
    '''
    :param msg:
    :return: 用于打印错误信息列表
    '''
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    '''
    用于判断输入的命令的格式是否规范
    :param args:
    :return:
    '''
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')#没有输入数据的路径
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')#没有提供输出数据路径
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')#没有指定输出路径。


def convert(def_path, caffemodel_path, data_output_path, code_output_path, phase):
    '''
    用于对输入的模型进行加载，和进行模型装换
    :param def_path: 输入的caffe网络结构文件prototxt格式
    :param caffemodel_path: 输入的caffe训练的到的模型文件caffemodel格式
    :param data_output_path: 输出的
    :param code_output_path:输出的tensorflow的网络结构代码
    :param phase:为网络的训练模式，test 或者是train
    :return:
    '''
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase)#加载网络结构和加载模型文件
        print_stderr('Converting data...')#开始转换数据
        if caffemodel_path is not None:
            data = transformer.transform_data()#转换权值和偏置数据
            print_stderr('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)#保存权值偏置数据
        if code_output_path:#转化为tensorflow对应网络我心代码
            print_stderr('Saving source...')
            with open(code_output_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())#保存网络模型代码
        print_stderr('Done.')#转换完成
    except KaffeError as err:#捕获异常
        fatal_error('Error encountered: {}'.format(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    #解析命令参数
    validate_arguments(args)#判断命令参数的是否有错误

    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path,
            args.phase)#加载模型，和转换模型


if __name__ == '__main__':
    main()
