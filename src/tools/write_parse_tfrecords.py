#!/usr/bin/env python3

from datadiff import diff
from utils import ask_yn
from glob import glob
import numpy as np
import os
from os import path as osp
import shutil
import tensorflow as tf
import yaml


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_feature(value, dtype):
    if dtype == "string":
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)
        return _bytes_feature([v.encode() for v in value.tolist()])
    elif dtype == "int64":
        return _int64_feature(value)
    else:
        raise ValueError("Unhandable dtype: %s" % dtype)


def write_tfrecord(data_dict, filename, verbose=False, dimension_dict=None):
    if dimension_dict is None:
        dimension_dict = dict()
    endswitch = ".tfrecords"
    if filename[-len(endswitch) :] != endswitch:
        filename += endswitch

    data_dict["filename"] = filename

    written_feature_format = {}
    feature = {}
    for key in data_dict:
        data = np.array(data_dict[key])
        dtype = data.dtype.name
        if dtype[:3] == "str":
            dtype = "string"
        written_feature_format[key] = dtype
        dims = data.shape
        if key in dimension_dict:
            fixed_dims = dimension_dict[key]
            assert len(fixed_dims) == len(dims)
            for d, fd in zip(dims, fixed_dims):
                if fd not in [-1, None]:
                    if d != fd:
                        raise ValueError(
                            "For %s array dimensions %s do not work with "
                            "provided dimensions %s!" % (key, dims, fixed_dims)
                        )
                    assert d == fd
        else:
            fixed_dims = [None for d in dims]
        if dtype != "string":
            data = data.tobytes()
        written_feature_format[key + "_raw"] = "string"
        if dtype == "string":
            feature[key + "_raw"] = get_feature(data, "string")
        else:
            feature[key + "_raw"] = _bytes_feature([data])
        for i, dim in enumerate(dims):
            if fixed_dims[i] in [-1, None]:
                written_feature_format[key + "_dim%d" % i] = -1
                feature[key + "_dim%d" % i] = _int64_feature([dim])
            else:
                written_feature_format[key + "_dim%d" % i] = fixed_dims[i]

    dirname = osp.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    format_filename = osp.join(dirname, "tfrecords_format.yml")
    if verbose:
        print("Writing", filename)
    if osp.exists(format_filename):
        with open(format_filename, "r") as fin:
            existing_feature_format = yaml.safe_load(fin)
        if existing_feature_format != written_feature_format:
            print(diff(existing_feature_format, written_feature_format))
            override = ask_yn(
                "New feature format detected! Do you want to override? "
                "Warning: Deletes complete content of folder!"
            )
            if not override:
                quit()
            else:
                shutil.rmtree(osp.dirname(format_filename))
                os.makedirs(osp.dirname(format_filename))
                with open(format_filename, "w") as fout:
                    fout.write(yaml.safe_dump(written_feature_format))
    else:
        with open(format_filename, "w") as fout:
            fout.write(yaml.safe_dump(written_feature_format))
    with tf.python_io.TFRecordWriter(filename) as writer:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


def tfrecord_parser(
    filenames,
    feature_format,
    keep_plain=False,
    parallel_prefetching=5,
    buffer_size=None,
    count=None,
):
    if buffer_size is None:
        buffer_size = 2 * len(filenames)
    multi_dims = {}
    encoding_features = {}
    for key in feature_format:
        if key + "_raw" in feature_format:
            dims = []
            while key + "_dim%d" % len(dims) in feature_format:
                dims += [feature_format[key + "_dim%d" % len(dims)]]
            multi_dims[key] = dims
        else:
            if key[-4:] == "_raw":
                encoding_features[key] = tf.FixedLenSequenceFeature(
                    [], getattr(tf, feature_format[key]), allow_missing=True
                )
            else:
                feature_type = feature_format[key]
                if "_dim" in key:
                    if feature_type == -1:
                        feature_type = "int64"
                    else:
                        continue
                encoding_features[key] = tf.FixedLenFeature(
                    [], getattr(tf, feature_type)
                )

    def parsing_func(proto):
        parsed = tf.parse_single_example(proto, encoding_features)
        for key in multi_dims:
            if feature_format[key] != "string":
                parsed[key] = tf.decode_raw(
                    parsed[key + "_raw"], out_type=feature_format[key]
                )
            else:
                parsed[key] = parsed[key + "_raw"]
            parsed[key] = tf.reshape(
                parsed[key],
                [
                    parsed[key + "_dim%d" % i] if dim == -1 else dim
                    for i, dim in enumerate(multi_dims[key])
                ],
            )
            del parsed[key + "_raw"]
            for i, dim in enumerate(multi_dims[key]):
                if dim == -1:
                    del parsed[key + "_dim%d" % i]
        return parsed

    if not keep_plain:
        np.random.shuffle(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if not keep_plain:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size, count=count)
        )
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=parallel_prefetching
        )
    )
    dataset = dataset.prefetch(parallel_prefetching)
    dataset = dataset.map(parsing_func)

    return dataset


def get_filenames_and_feature_format(dirname):
    filenames = sorted(glob(osp.join(dirname, "*.tfrecords")))
    with open(osp.join(dirname, "tfrecords_format.yml"), "r") as fin:
        feature_format = yaml.safe_load(fin)
    return filenames, feature_format


if __name__ == "__main__":
    dirname = "/tmp/testtfrecords"
    filename = osp.join(dirname, "test1")
    filename2 = osp.join(dirname, "test2")
    data_dict = dict(
        simple_string="hello world",
        list_of_strings=["hi", "ho0"],
        array_of_strings=np.array([["hi1", "ho2"], ["hi3", "ho40"]]),
        testfloat=1.0,
        testfloat32=np.array(1.0, dtype=np.float32),
        testfloat64=np.array(1.0, dtype=np.float64),
        testint=1,
        testint32=np.array(1, dtype=np.int32),
        testint64=np.array(1, dtype=np.int64),
        testfloatarray=np.eye(3),
        testintarray=np.eye(3, dtype=np.int32),
        testlistfloat=[1.0, 2.0],
        testlistint=[[1, 0], [2, 3]],
        testuint8=np.array(1, dtype=np.uint8),
        testuint16=np.array(1, dtype=np.uint16),
        testint16=np.array(1, dtype=np.int16),
        testemptyarray=np.empty((0, 18, 23)),
    )
    fixed_dims = dict(
        array_of_strings=(None, 2), testemptyarray=(None, 18, 22), testlistint=(2, -1)
    )
    write_tfrecord(data_dict, filename, dimension_dict=fixed_dims)
    data_dict["add_info"] = "info"
    write_tfrecord(data_dict, filename2, dimension_dict=fixed_dims)
    with open(osp.join(dirname, "tfrecords_format.yml"), "r") as fin:
        print("tfrecords_format.yml")
        print(yaml.safe_load(fin))
    fnames, feature_format = get_filenames_and_feature_format(dirname)
    tf_dataset = tfrecord_parser(fnames, feature_format)
    iterator_get_next = tf_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        for _ in range(len(fnames)):
            result = session.run(iterator_get_next)
            for k in result:
                if isinstance(result[k], np.ndarray):
                    print(
                        k,
                        iterator_get_next[k],
                        type(result[k]),
                        result[k].shape,
                        result[k],
                    )
                else:
                    print(k, iterator_get_next[k], type(result[k]), result[k])
