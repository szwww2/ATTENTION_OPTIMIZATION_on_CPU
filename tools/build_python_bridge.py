"""Build the allocating C API used by the PyTorch comparison (GCC)."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from build_integrated import ROOT, CXX, BASE_FLAGS


def main():
    folder = ROOT / "build" / "python_bridge"
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "build.json"
    record.write_text('{"status": "building"}\n', encoding="utf-8")
    sources = [
        (f"src/{name}.cpp", [])
        for name in ("input", "oracle_fp64", "full_matrix_scalar", "rowwise_scalar")
    ]
    sources += [
        ("src/simd_routes.cpp", ["-mavx2"]),
        ("src/omp_routes.cpp", ["-fopenmp"]),
        ("tools/attention_c_api.cpp", ["-fopenmp"]),
    ]
    objects, commands = [], []
    try:
        for source, extra in sources:
            obj = folder / (Path(source).stem + ".o")
            command = [CXX, *BASE_FLAGS, "-fPIC", *extra, "-c", source, "-o", str(obj)]
            subprocess.run(command, cwd=ROOT, check=True)
            commands.append(["g++", *BASE_FLAGS, "-fPIC", *extra, "-c", source])
            objects.append(str(obj))
        library = folder / ("gp8_attention.dll" if os.name == "nt" else "libgp8_attention.so")
        subprocess.run(
            [CXX, "-shared", *objects, "-fopenmp", "-o", str(library)], cwd=ROOT, check=True
        )
        tracked = [ROOT / source for source, _ in sources] + list((ROOT / "include").glob("*.h"))
        tracked += list((ROOT / "src").glob("*.h")) + [
            Path(__file__),
            ROOT / "tools/build_integrated.py",
        ]
        details = dict(
            status="passed",
            library=library.name,
            compiler=subprocess.check_output([CXX, "--version"], text=True).splitlines()[0],
            compile_commands=commands,
            link_flags=["-shared", "-fopenmp"],
            source_sha256={
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tracked
            },
            binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        )
        record.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        print("Built", library.name)
    except Exception:
        record.write_text('{"status": "failed"}\n', encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
