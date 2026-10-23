# Machine records for the 23 October study

These are the CPU topology and WMI cache records retained on the measurement machine. They are copied without changing their content.

- [cpu_topology.csv](cpu_topology.csv): output of [cpu_topology_win.cpp](../../tools/cpu_topology_win.cpp). In Windows processor group 0, CPU0–3 have efficiency class 1 and CPU4–15 class 0. Each listed core has one set bit, matching 16 physical/16 logical cores.
- [cache_wmi.json](cache_wmi.json): Win32_Processor Name, NumberOfCores, NumberOfLogicalProcessors, L2CacheSize and L3CacheSize. The cache fields are in KiB: 24576 KiB = 24 MiB L2; 18432 KiB = 18 MiB L3. These are WMI totals, not measured per-worker cache availability.

The [formal study](../controlled_20261023/README.md) used CPU0 for serial jobs and the first T CPU IDs for T-thread jobs. T8 therefore mixes the recorded classes. An affinity mask only limits eligible CPUs; it does not guarantee worker placement, exclusive access, cache residency or a fixed clock.

原机器的拓扑和缓存记录。核类别与缓存容量是机器属性记录，不是性能瓶颈的证明。
