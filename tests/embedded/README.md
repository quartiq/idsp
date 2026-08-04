# Embedded benchmarks for idsp

* Target: stm32h743
* load-to-ram, run-from-ram
* Caches disabled as ram is single cycle
* Simple yet accurate cycle counting for `dsp-process` implementors
* Large slice, small chunk, single sample and inplace processing
* `idsp` IIR biquads in various configurations
* `idsp` compared with `biquad-rs`
* Scalar-measurement Kalman filters: structural N=1 random walk, dense versus
  structural N=2 constant velocity, and dense versus direct-measurement N=4
  constant jerk, using floating- and fixed-point arithmetic
* `hbf` FIRs
* Tooling routines: `cossin`, `atan2`

## trig

```txt
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] identity              [19.0, 0.0, 0.0, 0.0, 0.0]
[INFO ] black_box             [25.0, 10.5, 9.516602, 10.5, 7.5878906]
[INFO ] cossin                [40.0, 28.0, 23.549805, 0.0, 0.0]
[INFO ] atan2                 [71.0, 54.25, 54.032227, 0.0, 0.0]
```

## hbf

```txt
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] fir evensym           [409.0, 187.75, 118.15039, 179.0, 117.16992]
[INFO ] hbf int8              [1287.0, 748.5, 528.08496, 0.0, 0.0]
[INFO ] hbf dec8              [1694.0, 752.5, 574.78516, 0.0, 0.0]
```

## biquad

```txt
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] idsp q32              [41.0, 15.75, 10.567383, 17.75, 8.597656]
[INFO ] idsp clamp q32        [46.0, 25.25, 16.073242, 25.25, 17.304688]
[INFO ] idsp wide q32         [54.0, 36.0, 22.060547, 32.0, 19.055664]
[INFO ] idsp dither q32       [45.0, 20.75, 11.8046875, 20.0, 11.807617]
[INFO ] idsp clamp wide q32   [63.0, 51.75, 31.086914, 43.25, 23.079102]
[INFO ] idsp clamp dither q32 [49.0, 29.5, 20.06543, 29.5, 22.55664]
[INFO ] idsp f32              [49.0, 20.25, 13.657227, 19.75, 13.404297]
[INFO ] biquad df1 f32        [56.0, 20.0, 13.658203, 20.75, 13.657227]
[INFO ] idsp df2t f32         [40.0, 18.75, 11.905273, 17.5, 12.024414]
[INFO ] biquad df2t f32       [45.0, 17.5, 11.152344, 16.5, 11.025391]
[INFO ] idsp clamp f32        [61.0, 42.5, 35.788086, 41.5, 35.036133]
[INFO ] idsp f64              [78.0, 55.5, 44.43457, 57.25, 44.42871]
[INFO ] biquad df1 f64        [80.0, 56.5, 44.436523, 57.5, 44.43457]
[INFO ] idsp df2t f64         [66.0, 50.25, 40.674805, 52.0, 40.67285]
[INFO ] biquad df2t f64       [62.0, 49.75, 41.301758, 52.25, 41.685547]
[INFO ] idsp clamp f64        [97.0, 71.75, 59.316406, 76.25, 59.311523]
[INFO ] idsp wdf-ca-7         [57.0, 52.75, 29.583984, 45.25, 25.080078]
[INFO ] idsp wdf-ca-19        [98.0, 84.25, 82.03418, 84.0, 81.02637]
[INFO ] idsp q16              [50.0, 18.0, 8.822266, 18.75, 7.9414062]
[INFO ] idsp q64              [177.0, 156.75, 137.57422, 157.25, 141.56738]
[INFO ] lowpass1              [27.0, 12.5, 7.544922, 11.25, 6.9189453]
[INFO ] lowpass2              [40.0, 17.75, 10.298828, 16.75, 10.549805]
[INFO ] idsp q32 Cascade4     [113.0, 90.5, 79.51367, 84.5, 76.03223]
[INFO ] idsp q32 x4           [85.0, 81.25, 36.689453, 58.5, 37.146484]
[INFO ] idsp f32 Cascade4     [102.0, 80.5, 47.629883, 82.0, 49.131836]
[INFO ] idsp f32 x4           [113.0, 82.25, 54.124023, 77.75, 54.10547]
[INFO ] biquad f32 x4         [114.0, 82.0, 54.123047, 77.25, 54.106445]
```

## kalman

```
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] kalman f32 n1 structu [63.0, 48.75, 43.291992, 47.5, 43.398438]
[INFO ] kalman f32 n2 dense   [158.0, 142.75, 143.03711, 143.5, 141.02832]
[INFO ] kalman f32 n2 structu [102.0, 80.75, 64.06543, 79.5, 66.55078]
[INFO ] kalman f32 n4 dense   [755.0, 741.5, 741.53613, 742.0, 737.5303]
[INFO ] kalman f32 n4 direct  [646.0, 635.75, 635.53613, 636.25, 633.5293]
[INFO ] kalman q32 n2 dense   [442.0, 441.25, 437.54102, 441.75, 437.5254]
[INFO ] kalman q32 n2 structu [370.0, 374.25, 370.1162, 375.5, 373.0293]
[INFO ] kalman q32 n4 dense   [1465.0, 1469.75, 1464.0459, 1468.0, 1464.0342]
[INFO ] kalman q32 n4 direct  [1261.0, 1264.75, 1263.043, 1264.5, 1261.0312]
```
