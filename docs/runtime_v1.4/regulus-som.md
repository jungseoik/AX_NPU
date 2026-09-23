<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/regulus-som.html (원본) -->

# REGULUS System-on-Module (SoM)

## Product Summary

![REGULUS System-on-Module (SoM)](../res/image/MLM-1_SoM_3.jpg)

**REGULUS** is a compact AI System-on-Chip (SoC) with **10 TOPS of computing power** at under **3W** of power, optimized for higher-performance on-device AI use cases in power- and space-constrained applications such as drones and security systems.

REGULUS comes pre-equipped on a system-on-module (SoM) board, making it a fully functioning AI PC in itself.

Programs for REGULUS are built in a cross-compilation environment and run on the module's ARM CPU. See [(Basic) Running Pre-Compiled ResNet50](tutorial_resnet50.md#for-regulus) for the cross-compilation workflow.

## Identifying the Target Device

REGULUS ships as different target devices — `regulus-ra` and `regulus-rb`. Run the following on the device to check its target device:

```bash
regulus-version 2>/dev/null | awk -F': *' '/^Target Device/{print $2; found=1} END{if(!found) print "regulus-ra"}'
```

You can also tell the target device apart by comparing the board's appearance with the images below.

### regulus-ra

![regulus-ra board](../res/image/regulus-ra.png){w=480px}

- SOM board: `REGULUS_SOM_Rev.02`
- Base board: `REGULUS_BASE_Rev.02` or `REGULUS_BASE_Rev.03`

### regulus-rb

![regulus-rb board](../res/image/regulus-rb.png){w=480px}

- Target board : `regulus-evk-rev01`
- SOM board: `REGULUS_SOM_REV01B` or `REGULUS_SOM_REV02B`
- Carrier board: `REGULUS_CARRIER_REV.01B`
