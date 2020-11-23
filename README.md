# astro_stacker.py

A simple python3 application to stack and optionally align raw images to reduce noise (using [OpenCV](https://opencv.org) and [RawPy](https://pypi.org/project/rawpy/)). Intended primarily for aligning astrophotography images taken with a consumer DSLR or mirrorless camera.

Images are stacked using 32-float accuracy and saved both as a 16bpc **linearly encoded** sRGB `.png` and a floating-point `.hdr` file.

## Usage

`./astro_stacker.py --help` provides detailed command line options:

```
astro_stacker

A simple command line tool to average (optionally with homography alignment) a bunch of raw files.

Usage:
  astro_stacker.py [options] INFILES... 
  astro_stacker.py -h | --help
  astro_stacker.py --version

Arguments:
  INFILES                   image files to stack

Options:
  -h --help                 show this help message and exit
  --version                 show version
  --debug=LEVEL             set debug level [default: 0]
  -a BOOL --align=BOOL      attempt to align the images using a homography [default: True]
  -t THR --threshold=THR    align using images thresholded at the percentile [default: 0.85]
  -c BOOL --cache=BOOL      cache the keypoints in the base frame [default: True]
  -p PAD --padding=PAD      add black padding (in fraction of width,height) around the image [default: 0.0]
  -m MASK --mask=MASK       filename of a mask frame (to generate alignment features only inside mask)
  -r BOOL --raw=BOOL        don't demosaic. keep as raw Bayer mosaic [default: False]
  -v BOOL --visible=BOOL    crop to the visible window [default: True]
  -b BASE --base=BASE       filename of base frame
  -d DARK --dark=DARK       filename of master dark frame
  -f FLAT --flat=FLAT       filename of master flat frame
  -o OUTFILE                set output path prefix (can include directory and filename prefix)

```

## Examples

### Creating average calibration frames (dark, bias, flat, or dark flat frames)

Average all `dark-frame*.dng` files in `input/` directory with no demosaicing (`-r 1`), no alignment (`-a 0`), no cropping to the visible window (`-v 0`), and output (`-o`) to a desired directory/filename prefix:
```
./astro_stacker.py -r 1 -a 0 -v 0 -o "output/master-dark-frame" "input/dark-frame"*.dng
```

### Align and stack a set of raw files

Align and stack all input `.dng` files without any calibration frames (uses automatic black-point). Uses the first image in alphabetical order as the base frame to align all other images to:
```
./astro_stacker.py -o "output/aligned-" "input/"*.dng
```

The above uses the first image in alphabetical order as the base frame to align all other images to. If you want to specify which image should be the base image, use the `-b` option:
```
./astro_stacker.py -b "input/align-all-to-this-frame.dng" -o "output/aligned-" "input/"*.dng
```
The base image can already be in the list of input files, or be an additional image.

Instead of using automatic black-point, subtract master dark frame (assumed to be a 16bpp linearly encoded grayscale PNG) from each image before aligning and stacking:
```
./astro_stacker.py -d "master-dark-frame.png" -o "output/aligned-" "input/"*.dng
```

Additionally use a master flat frame (again, assumed to be a 16bpp linearly encoded grayscale PNG) to compensate for vignetting and other non-uniformities like dust:
```
./astro_stacker.py -d "master-dark-frame.png" -f "master-flat-frame.png" -o "output/aligned-" "input/"*.dng
```


## Acknowledgements & Resources

Other projects and tutorials that I used as a reference when writing this code:
* [AKAZE matching OpenCV tutorial](https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html)
* [OpenCV feature matching tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
* [image-align-and-average GitHub project](https://github.com/michal2229/image-align-and-average)
* [image_stacking GitHub project](https://github.com/maitek/image_stacking)

Other useful information about astronomical image stacking:
* [Deep Sky Stacker documentation](http://deepskystacker.free.fr/english/theory.htm)
* [Deep Sky Stacker FAQ](http://deepskystacker.free.fr/english/theory.htm)
* [CCD Data Reduction Guide](https://mwcraig.github.io/ccd-as-book/)

