// Windows diagnostic: logical-processor masks and efficiency classes per core.
#include <windows.h>

#include <cstdio>
#include <vector>

int main() {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) return 1;
    std::vector<unsigned char> storage(bytes);
    auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &bytes)) return 2;

    std::printf("core,efficiency_class,group,mask_hex\n");
    unsigned core = 0;
    for (DWORD offset = 0; offset < bytes;) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data() + offset);
        if (info->Size == 0 || offset + info->Size > bytes) return 3;
        for (WORD group = 0; group < info->Processor.GroupCount; ++group) {
            const GROUP_AFFINITY& affinity = info->Processor.GroupMask[group];
            std::printf("%u,%u,%u,%llx\n", core,
                        static_cast<unsigned>(info->Processor.EfficiencyClass),
                        static_cast<unsigned>(affinity.Group),
                        static_cast<unsigned long long>(affinity.Mask));
        }
        offset += info->Size;
        ++core;
    }
    return 0;
}
