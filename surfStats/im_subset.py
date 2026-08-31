from osgeo import gdal
import numpy as np

# GDAL type name -> numpy dtype.  Keys are matched case-insensitively; see
# gdal.GetDataTypeName().  Int8 exists from GDAL 3.7 onward.
_GDAL_DTYPE={
        'Byte':np.ubyte,
        'Int8':np.int8,
        'Float16':np.float16,
        'Float32':np.float32, 'Float64':np.float64,
        'Int16':np.int16, 'UInt16':np.uint16,
        'Int32':np.int32, 'UInt32':np.uint32,
        'Int64':np.int64, 'UInt64':np.uint64}
_GDAL_DTYPE.update({key.lower():val for key, val in _GDAL_DTYPE.items()})

def gdal_dtype(type_name):
    try:
        return _GDAL_DTYPE[type_name.lower()]
    except KeyError:
        raise KeyError(
            f"no numpy dtype mapping for GDAL data type '{type_name}'; "
            f"known types are {sorted(set(_GDAL_DTYPE))}") from None

class virtualRaveledGrids(object):
    def __init__(self, x0, y0):
        self.x0=x0
        self.y0=y0
        self.shape=[len(y0), len(x0)]
    def __getitem__(self, index):
        r, c = np.unravel_index(index, self.shape)
        return self.x0[c], self.y0[r]

class im_subset:
    def __init__(self, c0, r0, Nc, Nr, source, pad_val=0, Bands=(1,2,3), stride=None, pad=None, no_edges=False):
        self.source=source
        self.c0=c0
        self.r0=r0
        self.Nc=Nc
        self.Nr=Nr
        self.z=[]
        if hasattr(self.source, 'level'): # if the level is zero, this is a copy of a file, if it's >0, it's a copy of a copy of a file
            self.level=self.source.level+1
        else:
            self.level=0
        self.Bands=Bands
        self.pad_val=pad_val
        if stride is not None:
            if not hasattr(stride,"__len__"):
                stride=np.array([stride, stride])
        else:
            stride=np.array([1, 1])
        if pad is None:
            pad=0
        if not hasattr(pad,"__len__"):
            pad=np.array([pad, pad])
        self.stride=stride
        self.pad=pad
        if not no_edges:
            x0=self.c0+np.arange(0, self.Nc, stride[0])
            y0=self.r0+np.arange(0, self.Nr, stride[1])
        else:
            x0=self.c0+np.arange(pad[0], self.Nc-pad[0], stride[0])
            y0=self.r0+np.arange(pad[1], self.Nr-pad[1], stride[1])
        self.xy0=virtualRaveledGrids(x0, y0)
        self.count=0

    def __getitem__(self, index):
        try:
            xy0=self.xy0[index]
        except ValueError:
            raise StopIteration
        self.setBounds( xy0[0]-self.pad[0], xy0[1]-self.pad[1],
                              self.stride[0]+2*self.pad[0], self.stride[1]+2*self.pad[1])
        self.copySubsetFrom()
        return self

    def setBounds(self, c0, r0, Nc, Nr, update=0):
        self.c0=int(c0)
        self.r0=int(r0)
        self.Nc=int(Nc)
        self.Nr=int(Nr)
        if update > 0:
            self.copySubsetFrom(pad_val=self.pad_val)

    def copySubsetFrom(self, pad_val=0):
        if hasattr(self.source, 'level'):  # copy data from another subset
            self.z = np.zeros((self.source.z.shape[0], self.Nr, self.Nc), self.source.z.dtype) + pad_val
            (sr0, sr1, dr0, dr1, vr)=match_range(self.source.r0, self.source.Nr, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(self.source.c0, self.source.Nc, self.c0, self.Nc)
            if (vr & vc):
                self.z[:, dr0:dr1, dc0:dc1]=self.source.z[:,sr0:sr1, sc0:sc1]
            self.level=self.source.level+1
        else:  # read data from a file
            band=self.source.GetRasterBand(int(self.Bands[0]))
            src_NB=self.source.RasterCount
            dt=gdal_dtype(gdal.GetDataTypeName(band.DataType))
            self.z=np.zeros((src_NB, self.Nr, self.Nc), dt)+pad_val
            (sr0, sr1, dr0, dr1, vr)=match_range(0, band.YSize, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(0, band.XSize, self.c0, self.Nc)
            if (vr & vc):
                a=self.source.ReadAsArray(int(sc0),  int(sr0), int(sc1-sc0), int(sr1-sr0))
                self.z[:, dr0:dr1, dc0:dc1]=a
            self.level=0


    def writeSubsetTo(self, bands, target, VERBOSE=False):
        if bands is None:
            bands=self.Bands
        if hasattr(target, 'level') and target.level > 0:
            print("copying into target raster")
            (sr0, sr1, dr0, dr1, vr)=match_range(target.source.r0, target.source.Nr, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(target.source.c0, target.source.Nc, self.c0, self.Nc)
            if (vr & vc):
                for b in bands:
                    target.source.z[b-1,sr0:sr1, sc0:sc1]=self.z[b-1, dr0:dr1, dc0:dc1]
        else:
            band=target.source.GetRasterBand(1)
            (sr0, sr1, dr0, dr1, vr)=match_range(0, band.YSize, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(0, band.XSize, self.c0, self.Nc)

            if (vr & vc):
                try:
                    if VERBOSE:
                        print( int(sc0), int(sr0))
                    for bb in (bands):
                        band=target.source.GetRasterBand(int(bb))
                        band.WriteArray( self.z[bb-1, dr0:dr1, dc0:dc1], int(sc0), int(sr0))
                except TypeError:
                     band=target.source.GetRasterBand(int(bands))
                     band.WriteArray( self.z[int(bands-1), dr0:dr1, dc0:dc1], int(sc0), int(sr0))

def match_range(s0, ns, d0, nd):
    i0 = max(s0, d0)
    i1 = min(s0+ns, d0+nd)
    si0=max(0, i0-s0)
    si1=min(ns, i1-s0)
    di0=max(0, i0-d0)
    di1=min(nd,i1-d0)
    any_valid=(di1>di0) & (si1 > si0)
    return (si0, si1, di0, di1, any_valid)
