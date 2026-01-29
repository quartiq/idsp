# Embedded benchmarks for idsp

* Target: stm32h743
* load-to-ram, run-from-ram
* Caches disabled as ram is single cycle
* Simple yet accurate cycle counting for `dsp-process` implementors
* Large slice, small chunk, single sample and inplace processing
* `idsp` IIR biquads in various configurations
* `idsp` compared with `biquad-rs`
* `hbf` FIRs
* Tooling routines: `cossin`, `atan2`

## trig

```txt
[INFO ] Starting
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] identity              [19.0, 0.0, 0.0, 0.0, 0.0]
[INFO ] black_box             [22.0, 10.5, 9.518555, 10.0, 7.591797]
[INFO ] cossin                [40.0, 27.75, 23.548828, 0.0, 0.0]
[INFO ] atan2                 [72.0, 61.5, 60.032227, 0.0, 0.0]
[INFO ] Done
```

## hbf

```txt
[INFO ] Starting
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] fir evensym           [434.0, 188.5, 115.47266, 174.5, 114.7832]
[INFO ] hbf int8              [1247.0, 816.75, 476.68652, 0.0, 0.0]
[INFO ] hbf dec8              [1684.0, 919.25, 545.5332, 0.0, 0.0]
[INFO ] Done
```

## biquad

```txt
[INFO ] Starting
[INFO ] Units: cycles per sample, chunk size: 4, slice size: 1024
[INFO ] Name                  [single, chunk, slice, chunk inplace, slice inplace]
[INFO ] idsp q32              [40.0, 15.25, 10.551758, 16.75, 8.682617]
[INFO ] idsp clamp q32        [48.0, 25.25, 16.140625, 25.75, 16.560547]
[INFO ] idsp wide q32         [56.0, 35.5, 22.061523, 32.25, 19.05664]
[INFO ] idsp dither q32       [47.0, 20.0, 11.0546875, 20.0, 11.0546875]
[INFO ] idsp clamp wide q32   [79.0, 66.5, 64.038086, 67.5, 63.026367]
[INFO ] idsp clamp dither q32 [59.0, 52.5, 39.03418, 47.75, 50.027344]
[INFO ] idsp f32              [43.0, 20.25, 13.65918, 20.0, 13.408203]
[INFO ] biquad df1 f32        [47.0, 20.25, 13.660156, 21.0, 13.658203]
[INFO ] idsp df2t f32         [42.0, 18.75, 11.90332, 18.5, 12.026367]
[INFO ] biquad df2t f32       [43.0, 17.25, 11.150391, 17.0, 11.027344]
[INFO ] idsp clamp f32        [63.0, 42.5, 35.788086, 42.5, 34.91504]
[INFO ] idsp f64              [79.0, 55.75, 44.433594, 54.5, 44.42578]
[INFO ] biquad df1 f64        [80.0, 55.75, 44.43457, 55.25, 44.427734]
[INFO ] idsp df2t f64         [64.0, 49.25, 40.677734, 49.5, 40.740234]
[INFO ] biquad df2t f64       [64.0, 49.5, 41.301758, 50.25, 41.791992]
[INFO ] idsp clamp f64        [97.0, 71.75, 59.30957, 73.25, 60.433594]
[INFO ] idsp wdf-ca-7         [55.0, 52.0, 29.585938, 45.75, 25.075195]
[INFO ] idsp wdf-ca-19        [91.0, 84.0, 82.03516, 81.5, 83.02832]
[INFO ] idsp q16              [40.0, 18.5, 8.822266, 19.25, 7.942383]
[INFO ] idsp q64              [170.0, 156.75, 137.5752, 154.0, 136.8711]
[INFO ] Done
```
