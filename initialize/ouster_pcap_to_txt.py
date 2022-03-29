from ouster import client
import numpy as np
from ouster import pcap
from contextlib import closing
from more_itertools import nth
import os
import sys
from itertools import islice

# with closing(client.Scans(source)) as scans:
#     scan = nth(scans, 50)
#     range_field = scan.field(client.ChanField.RANGE)
#     range_img = client.destagger(source.metadata, range_field)
#     xyzlut = client.XYZLut(source.metadata)
#     xyz = xyzlut(scan)
#     print(scan)

def pcap_to_txt(source: client.PacketSource,
                metadata: client.SensorInfo,
                start_idx: int = 0,
                end_idx: int = -1,
                txt_dir: str = ".",
                txt_base: str = "pcap_out",
                txt_ext: str = "txt"):
    """Write scans from a pcap to csv files (one per lidar scan).
    The number of saved lines per csv file is always H x W, which corresponds to
    a full 2D image representation of a lidar scan.
    Each line in a csv file is:
        TIMESTAMP, RANGE (mm), SIGNAL, NEAR_IR, REFLECTIVITY, X (mm), Y (mm), Z (mm)
    If ``csv_ext`` ends in ``.gz``, the file is automatically saved in
    compressed gzip format. :func:`.numpy.loadtxt` can be used to read gzipped
    files transparently back to :class:`.numpy.ndarray`.
    Args:
        pcap_path: path to the pcap file
        metadata_path: path to the .json with metadata (aka :class:`.SensorInfo`)
        start_idx: start index of scans to save from pcap to csv files
        end_idx: end index of scans to save from pcap to csv files
        csv_dir: path to the directory where csv files will be saved
        csv_base: string to use as the base of the filename for pcap output
        csv_ext: file extension to use, "csv" by default
    """

    # ensure that base csv_dir exists
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    field_names = 'TIMESTAMP (ns), RANGE (mm), SIGNAL, NEAR_IR, REFLECTIVITY, X (mm), Y (mm), Z (mm)'
    field_fmts = ['%.6f', '%.6f', '%.6f', '%d', '%.6f', '%d', '%d']

    # [doc-stag-pcap-to-csv]
    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    # if end_idx <= 0:
        # import more_itertools
        # end_idx = more_itertools.ilen(scans)
    if end_idx > 0:
        scans = islice(scans, end_idx)


    for idx, scan in enumerate(scans):

        if idx < start_idx:
            print(f'\rSkip {idx} frames...', end='', flush=True)
            continue
        # copy per-column timestamps for each channel
        col_timestamps = scan.header(client.ColHeader.TIMESTAMP)
        timestamps = np.tile(col_timestamps, (scan.h, 1))

        # grab channel data
        fields_values = [scan.field(ch) for ch in client.ChanField]

        # use integer mm to avoid loss of precision casting timestamps
        xyz = xyzlut(scan)
        channel = np.arange(xyz.shape[0]).reshape(-1,1)
        channel = np.repeat(channel, xyz.shape[1], axis=1)
        # get all data as one H x W x 8 int64 array for savetxt()
        frame = np.dstack((xyz, *fields_values, timestamps, channel)) 

        # not necessary, but output points in "image" vs. staggered order
        frame = client.destagger(metadata, frame)

        # write csv out to file
        # csv_path = os.path.join(csv_dir, f'{csv_base}_{idx:06d}.{csv_ext}')
        # save_path = os.path.join(txt_dir, f'{idx:06d}.txt')

        # header = '\n'.join([f'frame num: {idx}', field_names])

        # filter valid points
        frame_data = frame.reshape(-1, frame.shape[2])
        valid = []
        for i in range(frame_data.shape[0]):
            # range < 0.1m
            if frame_data[i, 3] < 100:
                continue
            valid.append(i)
        if len(valid) > 20000:
            save_frame = frame_data[valid][:, [0, 1, 2, 4, 7, 8, 5]]   #Point:0 Point:1 Point:2 Reflectivity Timestamp Channel
            save_frame[:, 4] = save_frame[:, 4] / 1e9       # nano second -> second
            order = np.argsort(save_frame[:, 4]).tolist()      # sort by time
            save_frame = save_frame[order] 
            # np.savetxt(save_path, save_frame, fmt=field_fmts)
            
            txt_name = f'{save_frame[0,4]:.3f}'.replace('.', '_') + '.txt'
            save_path = os.path.join(txt_dir, txt_name)
            np.savetxt(save_path, save_frame, fmt=field_fmts)
            
            print(f'\rwrite frame #{(idx)}/{end_idx - 1}, to file: {txt_name}', end="", flush=True)
            
        
if __name__ == '__main__':

    from _import_ import config_parser
    parser = config_parser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--pcap_path", type=str, default=None)
    args = parser.parse_args()

    print('Processing pcap...')
    print('File path: ', args.dataset_root)
    print('File name', args.data_name)
    print('start_idx', args.start_idx)
    print('end_idx', args.end_idx)
    if args.pcap_path:
        pcap_path = args.pcap_path
    else:
        pcap_path = os.path.join(args.dataset_root, args.data_name + '.pcap')

    metadata_path = 'live-1024x20.json'
    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())
    source = pcap.Pcap(pcap_path, metadata)

    dir_name = os.path.join(args.dataset_root, args.data_name + '_lidar_frames')
    pcap_to_txt(source, metadata, start_idx=args.start_idx, end_idx=args.end_idx, txt_dir=dir_name)