import os, logging
import tables
import numpy

def h5f_( fname, mode='r', title='', force=False):
    assert mode!='w' or (mode=='w' and force), 'should set force=True' 
    _path = os.path.dirname(fname)
    
    #logging.debug('path:>>>%s'%_path)
    if bool(_path) and not os.path.exists(_path):
        os.makedirs(_path)
    if mode=='r+' and not os.path.exists(fname):
        mode = 'w'
    
    if  hasattr(tables, 'open_file'):
        return tables.open_file(fname, mode=mode, title=title)
        
    obj = tables.openFile(fname, mode=mode, title=title)
    obj.get_node = obj.getNode
    obj.create_group = obj.createGroup
    obj.remove_node = obj.removeNode
    obj.create_carray = obj.createCArray
    obj.create_earray = obj.createEArray
    obj.create_vlarray = obj.createVLArray
    obj.create_table = obj.createTable

    return obj


def _dict2ttype(m):
    return dict([ (k,tables.Col.from_dtype(numpy.dtype(v))) for k,v in m.items() ])

def atom_(dt):
    if dt is 'obj':     return tables.ObjectAtom()
    if dt is 'bool':    return tables.BoolAtom()
    return tables.Atom.from_dtype(numpy.dtype(dt))

_FilterConf = {
    'zip':       tables.Filters(complevel=5, complib='zlib'),
    'blosc':     tables.Filters(complevel=3, complib='blosc'), 
    '_default':  tables.Filters(complevel=5, complib='zlib'),
}


_ByteAtom   = atom_('b')
_Int32Atom  = atom_('i4')
_BoolAtom   = atom_('bool')

def create_carray(h5f, where, name, dtype, shape, **args ): # columns are same type, shape of array is fixed.
    _filter  = args.get('filters', '_default')
    if isinstance(_filter, str):
        args['filters'] = _FilterConf[_filter]

    if isinstance(dtype, str):
        atom = atom_(dtype)
    elif isinstance(dtype, tables.atom.Atom):
        atom = dtype
    else:
        raise ValueError('dtype(%s) should be str or table.atom.Atom!'%dtype)

    return h5f.create_carray(where, name, atom, shape, **args)

def create_earray(h5f, where, name, dtype, **args ):    # columns are fixed type, rows aren't fixed 
    _filter  = args.get('filters', '_default')
    if isinstance(_filter, str):
        args['filters'] = _FilterConf[_filter]
    
    if isinstance(dtype, str):
        atom = atom_(dtype)
    elif isinstance(dtype, tables.atom.Atom):
        atom = dtype
    else:
        raise ValueError('dtype(%s) should be str or table.atom.Atom!'%dtype)

    return h5f.create_earray(where, name, atom, **args)


def create_vlarray(h5f, where, name, dtype, **args ):   # both of columns and rows aren't fixed 
    _filter  = args.get('filters', '_default')
    if isinstance(_filter, str):
        args['filters'] = _FilterConf[_filter]
    
    if isinstance(dtype, str):
        atom = atom_(dtype)
    elif isinstance(dtype, tables.atom.Atom):
        atom = dtype
    else:
        raise ValueError('dtype(%s) should be str or table.atom.Atom!'%dtype)

    return h5f.create_vlarray(where, name, atom, **args)


def create_table(h5f, where, name, dtype, **args):
    _filter  = args.get('filters', '_default')
    if isinstance(_filter, str):
        args['filters'] = _FilterConf[_filter]
    
    if isinstance(dtype, dict):
        dtype = _dict2ttype(dtype)     #dict will change the order of fields!, fix it later
    elif not isinstance(dtype, (tables.atom.Atom,numpy.dtype)):
        raise ValueError('dtype(%s) should be dict or table.atom.Atom!'%dtype)
    
    return h5f.create_table(where, name, dtype, **args)

#---
def rm(h5f, where, r=False):
    if not where in h5f: return
    
    pth,fn = os.path.dirname(where), os.path.basename(where)
    h5f.remove_node( pth, fn, r)


def mv(h5f, src, dest, replace=False, fields=None):
    assert src!=dest, 'src should not be same as dest!'
    if not dest in h5f:
        mkdir(h5f, dest)

    fields = fields or [e._v_name for e in h5f.list_nodes(src)]
    for k in fields:
        h5f.move_node(src, newparent=dest, name= k, overwrite=replace)        
    if not h5f.list_nodes(src):         rm(h5f, src)


def mkdir(h5f, path, mode='r+'):
    if path=='/' or (mode!='w' and (path in h5f)): return
    if mode=='w':    rm(h5f, path, r=True)
    
    pth,fn = os.path.dirname(path), os.path.basename(path)
    mkdir(h5f, pth)
    
    return h5f.create_group(pth, fn)
