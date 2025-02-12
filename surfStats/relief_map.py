#! /usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:41:53 2013

@author: ben
"""

try:
   from osgeo import gdal, gdalconst
except ImportError:
   import gdal
   from gdal import gdalconst

#gdal.SetCacheMax(10485760)

import argparse
import scipy.ndimage as snd
import numpy as np
from surfStats import im_subset
import scipy.stats as sps

import sys, os
import time
np.seterr(invalid='ignore')

def mask_pgc( this_bounds, mask, pgc_subs, dec):
    for key, sub in pgc_subs.items():
        sub.setBounds(*this_bounds, update=True)
    mask *= np.squeeze((pgc_subs['bitmask'].z == 0) | (pgc_subs['bitmask'].z == 2))
    if dec > 1:
        # skip erosion if the mask is all valid
        if not np.all(pgc_subs['matchtag'].z):
            mask *= snd.binary_erosion(
                snd.binary_erosion(np.squeeze(pgc_subs['matchtag'].z), np.ones((1, dec), dtype=bool), border_value=1),
                np.ones((dec,1), dtype=bool), border_value=1)
    else:
        mask &= pgc_subs['matchtag'].z

def main():
    parser = argparse.ArgumentParser(description='calculate scale map on the specified file.  Positional arguments give the input file and the maximum scale')
    parser.add_argument('input_file')
    parser.add_argument('--N', default=16., type=float, help='number of pixels over which to calculate the relief')
    parser.add_argument('--erode_scale', '-e', type=float, default=None)
    parser.add_argument('--mask_file', type=str)
    parser.add_argument('--pgc_masks', action='store_true', help='Use the PGC bitmask and matchtags to mask the input')
    parser.add_argument('--mask_values', type=float, nargs='+')
    parser.add_argument('--percent', type=float, default=2, help='percentile reported relative to the maximum and minimum (e.g. 2 gives the 2nd and 98th percentiles)')
    parser.add_argument('--out_label', type=str, help='add this string to the output file name')
    args=parser.parse_args()


    out_keys=['zmin','zmax', 'zmed','sigma']

    pct=[args.percent, 16., 50, 84, 100-args.percent]

    ds=gdal.Open(args.input_file)

    if args.mask_file is not None:
        mask_ds=gdal.Open(args.mask_file)
        mask_sub=im_subset(0, 0, 0, 0, mask_ds, pad_val=0, Bands=[1])
    out_base=os.path.splitext(args.input_file)[0];

    if args.out_label is not None:
        out_base += args.out_label

    out_files={}
    for key in out_keys:
        out_files[key]=out_base+f'_{key}_relief.tif'

    out_nodata=np.NaN

    print("working on %s, outfile is %s" % (args.input_file, out_files['zmed']) )

    for file in out_files.values():
        if os.path.exists(file):
            print("outfile %s exists, deleting" % file)
            os.remove(file)

    driver=gdal.GetDriverByName("GTiff")
    xform=np.array(ds.GetGeoTransform())

    band=ds.GetRasterBand(1)
    N=args.N
    dec=N/4;
    blocksize=np.minimum(4096, 8*N)
    inNoData=band.GetNoDataValue()
    if inNoData is None:
        inNoData = 0
    print("inNoData is %f" % (inNoData))
    print("stride is %d" % (blocksize-2*N))

    nX=band.XSize;
    nY=band.YSize;
    subs={}
    Dsets={}
    for key, file in out_files.items():
        Dsets[key] = driver.Create(file, int(nX/dec), int(nY/dec), 1,\
                                   gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
        subs[key]=im_subset(0, 0, int( nX/dec), int(nY/dec), Dsets[key], Bands=[1])

    if args.pgc_masks:
        pgc_subs={}
        for key, pad_val in zip(['matchtag','bitmask'], [1, 0]):
            sub_ds=gdal.Open(args.input_file.replace('_dem.tif','_'+key+'.tif'))
            pgc_subs[key] = im_subset(0, 0, nX, nY, sub_ds, pad_val=pad_val, Bands=[1])

    start_time=time.time()
    for out_ds in Dsets.values():
        band=out_ds.GetRasterBand(1)
        band.SetNoDataValue(out_nodata)

    if args.erode_scale is not None:
        erode_kernel=np.ones([1, 2*args.erode_scale]).astype('bool')

    start=time.time()
    dtime=0
    stride=blocksize-2*N
    print("nX_out=%f, nY_out=%f" % (int(nX/dec), int(nY/dec)))
    N_out_sub=int(blocksize/dec)-1
    for in_sub in im_subset(0, 0,  nX,  nY, ds, pad_val=0, Bands=[1], stride=blocksize-2*N, pad=N, no_edges=False):
        in_bounds = [in_sub.c0, in_sub.r0, in_sub.Nc, in_sub.Nr]
        out_bounds = [int((in_sub.c0+N/2)/dec), int((in_sub.r0+N/2)/dec), N_out_sub, N_out_sub]
        for sub in subs.values():
            sub.setBounds(*out_bounds)
            sub.z = np.zeros([len(sub.Bands), int( N_out_sub), int(N_out_sub)])+out_nodata

        sys.stdout.write("\r\b c0=%d/%d, r0=%d/%d, last dt=%f  " % \
                         (int(float(in_sub.c0)/float(stride)), \
                          int(float(nX)/float(stride)), \
                          int(float(in_sub.r0)/float(stride)), \
                          int(float(nY)/float(stride)), dtime))
        sys.stdout.flush()

        if args.mask_file is not None:
            mask_sub.setBounds(*in_bounds, update=True)
            in_sub.z.ravel()[~np.in1d(mask_sub.z.ravel(), args.mask_values)]=inNoData

        if args.pgc_masks:
            mask = np.ones_like(in_sub.z, dtype=bool)
            mask_pgc(in_bounds, mask, pgc_subs, int(dec))
            #if np.any(mask.ravel()==0) and not np.all((in_sub.z==inNoData) | (in_sub.z==0)):
            #    print('mask!')
            in_sub.z.ravel()[mask.ravel()==0] = inNoData

        if np.all(np.logical_or(in_sub.z == 0, np.logical_or(np.isnan(in_sub.z), in_sub.z==inNoData))):
            for sub in subs.values():
                sub.writeSubsetTo(None, sub)
            continue

        if args.erode_scale is not None:
            mask=np.logical_or(in_sub.z == 0., np.logical_or(np.isnan(in_sub.z), in_sub.z==inNoData))
            if np.any(mask):
                mask[0,:,:]=snd.morphology.binary_dilation(mask[0,:,:], structure=erode_kernel)
                mask[0,:,:]=snd.morphology.binary_dilation(mask[0,:,:], structure=erode_kernel.transpose())
            if np.all(mask):
                continue
            if args.landsat:
                in_sub.z=np.maximum(0, in_sub.z*2.e-5-0.1)

            in_sub.z[mask]=inNoData

        for this_sub in im_subset(in_sub.c0, in_sub.r0, blocksize, blocksize, in_sub, Bands=[[1]], stride=dec, pad=(N-dec)/2, no_edges=True):

            mask=np.logical_or(np.isnan(this_sub.z), this_sub.z==inNoData)
            if np.any(mask):
                continue

            imageData=np.float64(this_sub.z[0,:,:])
            r_out=int((this_sub.r0+N/2)/dec) - subs['zmed'].r0
            c_out=int((this_sub.c0+N/2)/dec) - subs['zmed'].c0
            dd = np.float64(this_sub.z[0,:,:])
            if np.sum(dd.ravel())==0:
                continue

            hmin, hm0, hmed, hm1, hmax = sps.scoreatpercentile(imageData.ravel(), pct)
            try:
                subs['zmin'].z[0, r_out, c_out]=hmed - hmin
            except Exception as e:
                print(e)
                pass
            subs['zmax'].z[0, r_out, c_out] = hmax - hmed
            subs['zmed'].z[0, r_out, c_out] = hmed
            subs['sigma'].z[0, r_out, c_out]=hm1-hm0

        for key, sub in subs.items():
            sub.writeSubsetTo(sub.Bands, sub)

        dtime=time.time()-start
        start=time.time()

    total_time=time.time()-start_time
    print("finished in %3.2f s" % (total_time))
    # test: moving the pixels by -dec/2
    xform[3]=xform[3]-xform[5]*dec/2
    xform[0]=xform[0]-xform[1]*dec/2
    xform[5]=xform[5]*dec
    xform[1]=xform[1]*dec

    for key, DS in Dsets.items():
        print(key)
        DS.SetGeoTransform(tuple(xform))
        DS.SetProjection(ds.GetProjection())

    sub=None
    del(Dsets)
    del(subs)
    import gc
    gc.collect()

if __name__=="__main__":
    main()
