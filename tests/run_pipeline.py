#!/usr/bin/env python
"""
Main execution script for Phase 1 CRM Analysis Pipeline
Run this script from PyCharm or command line to execute the complete pipeline

Usage:
    python scripts/run_pipeline.py                    # Run full pipeline
    python scripts/run_pipeline.py --stages collection cleaning  # Run specific stages
    python scripts/run_pipeline.py --resume-from engineering    # Resume from checkpoint
    python scripts/run_pipeline.py --help            # Show help
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.orchestrator import PipelineOrchestrator, PipelineStage
from config.settings import get_settings
from config.logging_config import LoggerSetup
from loguru import logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run CRM Data Analysis Pipeline - Phase 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  %(prog)s

  # Run only data collection and cleaning
  %(prog)s --stages collection cleaning

  # Resume from feature engineering
  %(prog)s --resume-from engineering

  # Run in production mode without plots
  %(prog)s --env production --skip-plots

  # Continue on errors
  %(prog)s --continue-on-error
        """
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['collection', 'cleaning', 'validation', 'engineering', 'eda', 'preparation'],
        help='Specific stages to run (default: all stages)'
    )

    parser.add_argument(
        '--resume-from',
        choices=['collection', 'cleaning', 'validation', 'engineering', 'eda', 'preparation'],
        help='Resume pipeline from specific stage using checkpoint'
    )

    parser.add_argument(
        '--env',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment to run in (default: development)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable data caching'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline even if a stage fails'
    )

    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip visualization generation (faster execution)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )

    parser.add_argument(
        '--data-sources',
        nargs='+',
        choices=['customers', 'transactions', 'interactions', 'marketing'],
        default=['customers', 'transactions', 'interactions', 'marketing'],
        help='Data sources to collect (default: all)'
    )

    parser.add_argument(
        '--target-column',
        type=str,
        default='churned',
        help='Target column for analysis (default: churned)'
    )

    parser.add_argument(
        '--export-results',
        action='store_true',
        help='Export all pipeline results to files'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )

    return parser.parse_args()


def map_stage_names(stage_names):
    """Map string stage names to PipelineStage enum"""
    mapping = {
        'collection': PipelineStage.DATA_COLLECTION,
        'cleaning': PipelineStage.DATA_CLEANING,
        'validation': PipelineStage.DATA_VALIDATION,
        'engineering': PipelineStage.FEATURE_ENGINEERING,
        'eda': PipelineStage.EXPLORATORY_ANALYSIS,
        'preparation': PipelineStage.MODEL_PREPARATION
    }

    if stage_names:
        return [mapping[name] for name in stage_names]
    return None


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        CRM DATA ANALYSIS PIPELINE - PHASE 1                                 ║
║        Industrial-Grade Customer Churn Analysis                             ║
║                                                                              ║
║        Version: 1.0.0                                                       ║
║        Framework: Industrial Data Pipeline                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_configuration(args, settings):
    """Print pipeline configuration"""
    print("\n📋 PIPELINE CONFIGURATION")
    print("=" * 50)
    print(f"Environment:        {args.env}")
    print(f"Target Column:      {args.target_column}")
    print(f"Data Sources:       {', '.join(args.data_sources)}")
    print(f"Cache Enabled:      {'No' if args.no_cache else 'Yes'}")
    print(f"Generate Plots:     {'No' if args.skip_plots else 'Yes'}")
    print(f"Continue on Error:  {'Yes' if args.continue_on_error else 'No'}")

    if args.stages:
        print(f"Stages to Run:      {', '.join(args.stages)}")
    elif args.resume_from:
        print(f"Resume From:        {args.resume_from}")
    else:
        print(f"Stages to Run:      All stages")

    print(f"Project Root:       {project_root}")
    print(f"Data Directory:     {settings.paths.DATA_DIR}")
    print(f"Reports Directory:  {settings.paths.REPORTS_DIR}")
    print("=" * 50 + "\n")


def run_dry_run(args):
    """Show what would be executed without running"""
    print("\n🔍 DRY RUN MODE - No execution will occur")
    print("=" * 50)

    stages = map_stage_names(args.stages) if args.stages else [
        PipelineStage.DATA_COLLECTION,
        PipelineStage.DATA_CLEANING,
        PipelineStage.DATA_VALIDATION,
        PipelineStage.FEATURE_ENGINEERING,
        PipelineStage.EXPLORATORY_ANALYSIS,
        PipelineStage.MODEL_PREPARATION
    ]

    print("\nStages that would be executed:")
    for i, stage in enumerate(stages, 1):
        print(f"  {i}. {stage.value}")

    print("\nParameters that would be used:")
    print(f"  - Data sources: {args.data_sources}")
    print(f"  - Target column: {args.target_column}")
    print(f"  - Use cache: {not args.no_cache}")
    print(f"  - Generate plots: {not args.skip_plots}")
    print(f"  - Continue on error: {args.continue_on_error}")

    print("\nExpected outputs:")
    print("  - data/raw/: Raw data files")
    print("  - data/processed/: Processed features")
    print("  - reports/: Analysis reports")
    print("  - reports/figures/: Visualizations")
    print("  - logs/: Execution logs")

    print("\n✓ Dry run completed. Add --dry-run=false to execute.")
    return 0


def main():
    """Main execution function"""

    # Parse arguments
    args = parse_arguments()

    # Print banner
    print_banner()

    # Set environment
    os.environ['ENV'] = args.env

    # Initialize settings
    settings = get_settings()

    # Load custom config if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            logger.info(f"Loading custom configuration from {config_path}")
            # Load custom settings here
        else:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

    # Setup logging
    log_level = "DEBUG" if args.verbose else settings.log_level
    log_setup = LoggerSetup(
        log_dir=settings.paths.LOGS_DIR,
        log_level=log_level
    )

    # Print configuration
    print_configuration(args, settings)

    # Dry run mode
    if args.dry_run:
        return run_dry_run(args)

    logger.info("=" * 80)
    logger.info("STARTING PIPELINE EXECUTION")
    logger.info("=" * 80)
    logger.info(f"Start Time: {datetime.now()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info("=" * 80)

    try:
        # Initialize pipeline orchestrator
        logger.info("Initializing Pipeline Orchestrator...")
        orchestrator = PipelineOrchestrator(settings)

        # Prepare pipeline parameters
        pipeline_params = {
            'use_cache': not args.no_cache,
            'continue_on_error': args.continue_on_error,
            'generate_plots': not args.skip_plots,
            'deep_clean': True,
            'target_column': args.target_column,
            'data_sources': args.data_sources
        }

        # Map stages
        stages = map_stage_names(args.stages) if args.stages else None
        resume_from = map_stage_names([args.resume_from])[0] if args.resume_from else None

        # Run pipeline
        logger.info("Executing pipeline...")
        results = orchestrator.run_pipeline(
            stages=stages,
            resume_from=resume_from,
            **pipeline_params
        )

        # Export results if requested
        if args.export_results:
            logger.info("Exporting pipeline results...")
            orchestrator.export_results()

        # Print summary
        print("\n" + "=" * 80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 80)

        summary = orchestrator.get_pipeline_summary()

        print(f"\nTotal Stages:     {summary['total_stages']}")
        print(f"Successful:       {summary['successful_stages']} ✓")
        print(f"Failed:           {summary['failed_stages']} ✗")
        print(f"Total Time:       {summary['total_execution_time']:.2f} seconds")

        print("\n📋 Stage Results:")
        print("-" * 50)
        for stage_name, stage_info in summary['stages'].items():
            status = "✓" if stage_info['success'] else "✗"
            print(f"{status} {stage_name:20s} | Time: {stage_info['execution_time']:6.2f}s")
            if stage_info['error']:
                print(f"  └─ Error: {stage_info['error']}")

        # Determine exit code
        exit_code = 0 if summary['failed_stages'] == 0 else 1

        print("\n" + "=" * 80)
        if exit_code == 0:
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\n📁 Output Locations:")
            print(f"  • Processed Data:  {settings.paths.PROCESSED_DATA_DIR}")
            print(f"  • Reports:         {settings.paths.REPORTS_DIR}")
            print(f"  • Visualizations:  {settings.paths.FIGURES_DIR}")
            print(f"  • Logs:           {settings.paths.LOGS_DIR}")

            print("\n🎯 Key Files Generated:")
            master_features_path = settings.paths.PROCESSED_DATA_DIR / "master_features.parquet"
            if master_features_path.exists():
                print(f"  • Master Features: {master_features_path}")

            eda_report_path = settings.paths.REPORTS_DIR / "eda_report.json"
            if eda_report_path.exists():
                print(f"  • EDA Report:      {eda_report_path}")

            pipeline_report_path = settings.paths.REPORTS_DIR / "pipeline_report.json"
            if pipeline_report_path.exists():
                print(f"  • Pipeline Report: {pipeline_report_path}")

            print("\n📈 Next Steps:")
            print("  1. Review the EDA report and visualizations")
            print("  2. Share findings with stakeholders")
            print("  3. Proceed to Phase 2: Model Development")
            print("     - Target: 85% accuracy for churn prediction")
            print("     - Use processed data from Phase 1")
            print("\n💡 Tip: Open reports/figures/interactive_dashboard.html in browser")
        else:
            print("❌ PIPELINE FAILED")
            print("=" * 80)
            print("\n⚠️  Some stages failed. Please check:")
            print(f"  • Error logs:      {settings.paths.LOGS_DIR}/errors.log")
            print(f"  • Pipeline report: {settings.paths.REPORTS_DIR}/pipeline_report.json")
            print("\n💡 Tips for troubleshooting:")
            print("  • Use --verbose flag for detailed logging")
            print("  • Check data quality in failed stages")
            print("  • Use --continue-on-error to skip failing stages")
            print("  • Resume from last successful stage with --resume-from")

        print("\n" + "=" * 80)
        print(f"End Time: {datetime.now()}")
        print("=" * 80 + "\n")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        print("\nPipeline interrupted. You can resume from the last checkpoint.")
        sys.exit(130)

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        logger.exception(e)
        print("\n" + "=" * 80)
        print("❌ FATAL ERROR")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print(f"\nCheck logs for details: {settings.paths.LOGS_DIR}/errors.log")
        print("\n💡 Common issues:")
        print("  • Missing dependencies: pip install -r requirements.txt")
        print("  • Invalid configuration: Check .env file")
        print("  • Insufficient memory: Reduce chunk_size in settings")
        print("  • Data access issues: Check database connection")
        sys.exit(1)


if __name__ == "__main__":
    main()