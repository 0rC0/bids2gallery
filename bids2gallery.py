
import nibabel as nib
import glob
from matplotlib import pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage


cmap=LinearSegmentedColormap.from_list('triple_cmap', colors= [(1, 0, 0), (0, 1, 0), (0,0,1)])

html_begin='''
<html>
<head>
<style>
body {
    background-color: #a9a9a9;
}
</style> 
</head>
<body>
'''

html_end='''
</body>
</html>'''

def sub2info(i):
    return i.split('/')[-4], i.split('/')[-3]

def sub2roi(i):
    return i.replace('/ICH/', '/ICH/derivatives/groundtruth_ICH_IVH_PHE_masks/').replace('_space-native_desc-CT.nii.gz', '_space-native_desc-fusion-ICH-PHE-IVH_mask.nii.gz')

def subs2html(subs, fname = 'gallery.html'):
    outs, errors = gen_imgs(subs)
    with open(fname, 'w') as html:
        html.write(html_begin)
        for i in outs:
            html.write('<h2>{} {}</h2>'.format(i['sid'], i['ses']))
            html.write('<img src="{img}" alt="{img}"></img>'.format(img=i['img']))
        if len(errors) > 0:
            html.write('<ul>')
            for i in errors:
                html.write('<li>{}</li>'.format(i))
            html.write('</ul>')
        html.write(html_end)
    return fname



def gen_imgs(subs):
    outs = []
    errors = []
    for sub in subs:
        try:
            sid, ses = sub2info(sub)
            img_arr = nib.load(sub).get_fdata()
            mask_arr = nib.load(sub2roi(sub)).get_fdata().astype(np.int64)
            img = get_plot(img_arr, mask_arr)
            img_name = './imgs/{}_{}.png'.format(sid, ses)
            img.savefig(img_name)
            outs.append({'sid': sid, 'ses': ses, 'img': img_name})
            print(outs[-1])
        except:
            errors.append(sub)
    return outs, errors

def get_cog(mask_arr):
    lbl = ndimage.label(mask_arr)[0]
    cog = ndimage.measurements.center_of_mass(mask_arr, lbl, [1, 2, 3])
    return cog

def get_plot(img_arr, mask_arr):
    cog = get_cog(mask_arr)
    c = int(cog[0][2])
    diff = int(c * 0.1)
    l = list(range(c - 3 * diff, c + 3 * diff, diff))
    fig, ax = plt.subplots(2, len(l))
    for n, y in enumerate(l):
        unmasked_mask = mask_arr[:, :, y]
        masked_mask = np.ma.masked_where(unmasked_mask == 0, unmasked_mask)
        ax[0, n].set_xticklabels([])
        ax[0, n].set_xticks([])
        ax[0, n].set_yticklabels([])
        ax[0, n].set_yticks([])

        ax[0, n].imshow(img_arr[:, :, y], cmap='gray', alpha=1, vmin=0, vmax=80)
        ax[1, n].set_xticklabels([])
        ax[1, n].set_xticks([])
        ax[1, n].set_yticklabels([])
        ax[1, n].set_yticks([])

        ax[1, n].imshow(img_arr[:, :, y], cmap='gray', alpha=1, vmin=0, vmax=80)
        ax[1, n].imshow(masked_mask, cmap=cmap, alpha=0.7)

    plt.gcf().set_size_inches(12, 5)
    return plt
