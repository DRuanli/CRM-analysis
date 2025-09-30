"""
Main execution script for Phase 1 CRM Analysis Pipeline
Run this script from PyCharm or command line to execute the complete pipeline
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
        description='Run CRM Data Analysis Pipeline - Phase 1'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['collection', 'cleaning', 'validation', 'engineering', 'eda', 'preparation'],
        help='Specific stages to run (default: all)'
    )

    parser.add_argument(
        '--resume-from',
        choices=['collection', 'cleaning', 'validation', 'engineering', 'eda', 'preparation'],
        help='Resume pipeline from specific stage'
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
        help='Skip visualization generation'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
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


def main():
    """Main execution function"""

    # Parse arguments
    args = parse_arguments()

    # Set environment
    os.environ['ENV'] = args.env

    # Initialize settings
    settings = get_settings()

    # Setup logging
    log_setup = LoggerSetup(
        log_dir=settings.paths.LOGS_DIR,
        log_level=settings.log_level
    )

    logger.info("=" * 80)
    logger.info("CRM DATA ANALYSIS PIPELINE - PHASE 1")
    logger.info("=" * 80)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Start Time: {datetime.now()}")
    logger.info("=" * 80)

    try:
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator(settings)

        # Prepare pipeline parameters
        pipeline_params = {
            'use_cache': not args.no_cache,
            'continue_on_error': args.continue_on_error,
            'generate_plots': not args.skip_plots,
            'deep_clean': True,
            'target_column': 'churned'
        }

        # Map stages
        stages = map_stage_names(args.stages) if args.stages else None
        resume_from = map_stage_names([args.resume_from])[0] if args.resume_from else None

        # Run pipeline
        results = orchestrator.run_pipeline(
            stages=stages,
            resume_from=resume_from,
            **pipeline_params
        )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)

        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful

        logger.info(f"Total Stages: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")

        # Print detailed results
        for stage, result in results.items():
            status = "✓" if result.success else "✗"
            logger.info(f"{status} {stage.value}: {result.execution_time:.2f}s")
            if result.error:
                logger.error(f"  Error: {result.error}")

        # Exit code based on success
        exit_code = 0 if failed == 0 else 1

        logger.info("\n" + "=" * 80)
        if exit_code == 0:
            logger.success("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("Check the following directories for outputs:")
            logger.info(f"  - Processed Data: {settings.paths.PROCESSED_DATA_DIR}")
            logger.info(f"  - Reports: {settings.paths.REPORTS_DIR}")
            logger.info(f"  - Visualizations: {settings.paths.FIGURES_DIR}")
            logger.info("\nNext Steps:")
            logger.info("  1. Review the EDA report and visualizations")
            logger.info("  2. Share findings with stakeholders")
            logger.info("  3. Proceed to Phase 2: Model Development")
        else:
            logger.error("PIPELINE FAILED - Please check the logs for details")

        logger.info("=" * 80)

        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
