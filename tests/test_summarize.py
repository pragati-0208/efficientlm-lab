from efficientlm.summarize import aggregate, load_runs, to_markdown


def test_aggregate_mean_std(fake_results):
    agg = aggregate(load_runs(fake_results))
    assert set(agg) == {"full_ft", "lora_r8", "student_kd"}
    mean, std = agg["full_ft"]["test_acc"]
    assert abs(mean - 0.907) < 1e-9 and std > 0
    assert agg["full_ft"]["n_seeds"] == 2


def test_markdown_sorted_best_first(fake_results):
    md = to_markdown(aggregate(load_runs(fake_results)))
    rows = [l for l in md.splitlines() if l.startswith("| ") and "method" not in l and "---" not in l]
    assert rows[0].startswith("| full_ft")
