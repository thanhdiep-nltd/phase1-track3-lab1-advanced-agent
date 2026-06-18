from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_mini.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    
    react_records = []
    reflexion_records = []
    
    out_path = Path(out_dir)
    mode_str = "mock" if os.environ.get("MOCK_MODE", "false").strip().lower() == "true" else "live"
    
    print(f"\n[bold blue]=== Starting Benchmark ===[/bold blue]")
    print(f"Mode: [bold]{mode_str}[/bold]")
    print(f"Dataset: [bold]{dataset}[/bold] ({len(examples)} examples)")
    print(f"Output Directory: [bold]{out_dir}[/bold]\n")
    
    # Run ReAct Agent
    print("[bold yellow]>>> Running ReAct Agent[/bold yellow]")
    for idx, example in enumerate(examples):
        print(f"  [yellow]ReAct[/yellow] | [{idx+1}/{len(examples)}] Processing {example.qid}...")
        record = react.run(example)
        react_records.append(record)
    print("[bold green]>>> ReAct Agent complete![/bold green]\n")
        
    # Run Reflexion Agent
    print("[bold cyan]>>> Running Reflexion Agent[/bold cyan]")
    for idx, example in enumerate(examples):
        print(f"  [cyan]Reflexion[/cyan] | [{idx+1}/{len(examples)}] Processing {example.qid}...")
        record = reflexion.run(example)
        reflexion_records.append(record)
    print("[bold green]>>> Reflexion Agent complete![/bold green]\n")
            
    all_records = react_records + reflexion_records
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode_str)
    json_path, md_path = save_report(report, out_path)
    
    print(f"[bold green][OK] Benchmark execution complete![/bold green]")
    print(f"Saved JSON report: [bold]{json_path}[/bold]")
    print(f"Saved Markdown report: [bold]{md_path}[/bold]\n")
    print("[bold]Summary Statistics:[/bold]")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
