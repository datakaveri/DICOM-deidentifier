import os

import pytest

import runtime


@pytest.fixture
def machine(monkeypatch):
    """Fake /proc and /sys files plus CPU affinity; returns the file dict."""
    files = {}
    monkeypatch.setattr(runtime, "_read", lambda path: files.get(path))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)))
    for var in (
        "SKALD_WORKERS",
        "SKALD_THREADS_PER_WORKER",
        "SKALD_MODEL_MEMORY_MB",
        "SKALD_WORKER_MEMORY_MB",
        "SKALD_FILE_TIMEOUT_S",
    ):
        monkeypatch.delenv(var, raising=False)
    return files


def meminfo(mb):
    return f"MemTotal: 99999999 kB\nMemAvailable: {mb * 1024} kB\n"


def test_cpu_budget_uses_affinity_without_quota(machine):
    assert runtime.cpu_budget() == 12


def test_cpu_budget_honours_cgroup_v2_quota(machine):
    machine["/sys/fs/cgroup/cpu.max"] = "250000 100000"  # docker --cpus 2.5
    assert runtime.cpu_budget() == 3


def test_cpu_budget_ignores_unlimited_v2_quota(machine):
    machine["/sys/fs/cgroup/cpu.max"] = "max 100000"
    assert runtime.cpu_budget() == 12


def test_cpu_budget_reads_the_process_cgroup_on_a_bare_vm(machine):
    machine["/proc/self/cgroup"] = "0::/system.slice/skald.service\n"
    machine["/sys/fs/cgroup/system.slice/skald.service/cpu.max"] = "400000 100000"
    assert runtime.cpu_budget() == 4


def test_cpu_budget_honours_cgroup_v1_quota(machine):
    machine["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"] = "200000"
    machine["/sys/fs/cgroup/cpu/cpu.cfs_period_us"] = "100000"
    assert runtime.cpu_budget() == 2


def test_memory_budget_takes_the_smaller_of_host_and_cgroup(machine):
    machine["/proc/meminfo"] = meminfo(16000)
    machine["/sys/fs/cgroup/memory.max"] = str(8 * 1024**3)
    machine["/sys/fs/cgroup/memory.current"] = str(1 * 1024**3)
    assert runtime.memory_budget_mb() == 7 * 1024


def test_memory_budget_ignores_unlimited_v1(machine):
    machine["/proc/meminfo"] = meminfo(5000)
    machine["/sys/fs/cgroup/memory/memory.limit_in_bytes"] = str(2**63 - 4096)
    assert runtime.memory_budget_mb() == 5000


def test_memory_budget_unknown(machine):
    assert runtime.memory_budget_mb() is None


def test_plan_twelve_cpus_is_one_process_with_all_threads(machine):
    machine["/proc/meminfo"] = meminfo(64000)
    p = runtime.plan(100)
    assert (p.workers, p.threads) == (1, 12)


def test_plan_large_machine_splits_into_workers(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(32)))
    machine["/proc/meminfo"] = meminfo(64000)
    p = runtime.plan(100)
    assert (p.workers, p.threads) == (4, 8)


def test_plan_memory_caps_workers(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(32)))
    machine["/proc/meminfo"] = meminfo(4500 + 512 + 2 * 1500)
    p = runtime.plan(100)
    assert (p.workers, p.threads) == (2, 16)


def test_plan_unknown_memory_stays_single_process(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)))
    p = runtime.plan(100)
    assert (p.workers, p.threads) == (1, 64)


def test_plan_never_more_workers_than_files(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)))
    machine["/proc/meminfo"] = meminfo(256000)
    assert runtime.plan(2).workers == 2


def test_plan_env_overrides(machine, monkeypatch):
    monkeypatch.setenv("SKALD_WORKERS", "3")
    monkeypatch.setenv("SKALD_THREADS_PER_WORKER", "2")
    p = runtime.plan(100)
    assert (p.workers, p.threads) == (3, 2)


def test_plan_rejects_bad_override(machine, monkeypatch):
    monkeypatch.setenv("SKALD_WORKERS", "0")
    with pytest.raises(SystemExit):
        runtime.plan(10)


def test_configure_keeps_parent_single_threaded_when_forking(machine, monkeypatch):
    # An OpenMP pool started in the parent deadlocks forked children.
    monkeypatch.setenv("SKALD_WORKERS", "2")
    for var in runtime._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    p = runtime.configure(10)
    assert p.threads == 6
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_configure_single_process_uses_all_threads(machine, monkeypatch):
    for var in runtime._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    runtime.configure(10)
    assert os.environ["OMP_NUM_THREADS"] == "12"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
