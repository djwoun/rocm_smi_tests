## `amdsmi_get_gpu_cache_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `info`: `amdsmi_gpu_cache_info_t *`

```c
amdsmi_status_t amdsmi_get_gpu_cache_info(amdsmi_processor_handle processor_handle, amdsmi_gpu_cache_info_t * info);
```

## `amdsmi_get_gpu_mem_overdrive_level`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `od`: `uint32_t *`

```c
amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t * od);
```

## `amdsmi_get_gpu_od_volt_curve_regions`

- **Returns:** `amdsmi_status_t`
- **Parameters (3):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `num_regions`: `uint32_t *`
  - `buffer`: `amdsmi_freq_volt_region_t *`

```c
amdsmi_status_t amdsmi_get_gpu_od_volt_curve_regions(amdsmi_processor_handle processor_handle, uint32_t * num_regions, amdsmi_freq_volt_region_t * buffer);
```

## `amdsmi_get_gpu_od_volt_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `odv`: `amdsmi_od_volt_freq_data_t *`

```c
amdsmi_status_t amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle processor_handle, amdsmi_od_volt_freq_data_t * odv);
```

## `amdsmi_get_gpu_overdrive_level`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `od`: `uint32_t *`

```c
amdsmi_status_t amdsmi_get_gpu_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t * od);
```

## `amdsmi_get_gpu_perf_level`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `perf`: `amdsmi_dev_perf_level_t *`

```c
amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t * perf);
```

## `amdsmi_get_gpu_pm_metrics_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (3):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `pm_metrics`: `amdsmi_name_value_t**`
  - `num_of_metrics`: `uint32_t *`

```c
amdsmi_status_t amdsmi_get_gpu_pm_metrics_info(amdsmi_processor_handle processor_handle, amdsmi_name_value_t** pm_metrics, uint32_t * num_of_metrics);

```

## `amdsmi_get_gpu_ras_feature_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `ras_feature`: `amdsmi_ras_feature_t *`

```c
amdsmi_status_t amdsmi_get_gpu_ras_feature_info(amdsmi_processor_handle processor_handle, amdsmi_ras_feature_t * ras_feature);
```

## `amdsmi_get_gpu_reg_table_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (4):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `reg_type`: `amdsmi_reg_type_t`
  - `reg_metrics`: `amdsmi_name_value_t**`
  - `num_of_metrics`: `uint32_t *`

```c
amdsmi_status_t amdsmi_get_gpu_reg_table_info(amdsmi_processor_handle processor_handle, amdsmi_reg_type_t reg_type, amdsmi_name_value_t** reg_metrics, uint32_t * num_of_metrics);
```

## `amdsmi_get_gpu_volt_metric`

- **Returns:** `amdsmi_status_t`
- **Parameters (4):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `sensor_type`: `amdsmi_voltage_type_t`
  - `metric`: `amdsmi_voltage_metric_t`
  - `voltage`: `int64_t *`

```c
amdsmi_status_t amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle, amdsmi_voltage_type_t sensor_type, amdsmi_voltage_metric_t metric, int64_t * voltage);
```

## `amdsmi_get_gpu_vram_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `info`: `amdsmi_vram_info_t *`

```c
amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle, amdsmi_vram_info_t * info);
```

## `amdsmi_get_hsmp_metrics_table`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `metrics_table`: `amdsmi_hsmp_metrics_table_t *`

```c
amdsmi_status_t amdsmi_get_hsmp_metrics_table(amdsmi_processor_handle processor_handle, amdsmi_hsmp_metrics_table_t * metrics_table);
```

## `amdsmi_get_hsmp_metrics_table_version`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `metrics_version`: `uint32_t *`

```c
amdsmi_status_t amdsmi_get_hsmp_metrics_table_version(amdsmi_processor_handle processor_handle, uint32_t * metrics_version);
```

## `amdsmi_get_pcie_info`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `info`: `amdsmi_pcie_info_t *`

```c
amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t * info);
```

## `amdsmi_get_processor_count_from_handles`

- **Returns:** `amdsmi_status_t`
- **Parameters (5):**
  - `processor_handles`: `amdsmi_processor_handle*`
  - `processor_count`: `uint32_t*`
  - `nr_cpusockets`: `uint32_t*`
  - `nr_cpucores`: `uint32_t*`
  - `nr_gpus`: `uint32_t*`

```c
amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor_handle, amdsmi_dpm_policy_t* policy);
```

## `amdsmi_get_xgmi_plpd`

- **Returns:** `amdsmi_status_t`
- **Parameters (2):**
  - `processor_handle`: `amdsmi_processor_handle`
  - `xgmi_plpd`: `amdsmi_dpm_policy_t*`

```c
amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle processor_handle, amdsmi_dpm_policy_t* xgmi_plpd);
```
