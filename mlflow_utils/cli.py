"""
MLflow CLI for model management.

Provides commands for:
- rollback: Revert to a previous model version
- list-versions: Show model version history
- promote: Manually promote a model version
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = str(PROJECT_ROOT / "mlruns")


def get_client(tracking_uri: Optional[str] = None) -> MlflowClient:
    """Get MLflow client with configured tracking URI."""
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


# ============================================================================
# Commands
# ============================================================================


def cmd_list_versions(
    model_name: str,
    tracking_uri: Optional[str] = None,
    show_archived: bool = False,
) -> List[ModelVersion]:
    """
    List all versions of a registered model.
    
    Args:
        model_name: Name of the registered model.
        tracking_uri: MLflow tracking URI.
        show_archived: Include archived versions.
        
    Returns:
        List of ModelVersion objects.
    """
    client = get_client(tracking_uri)
    
    try:
        if show_archived:
            stages = ["None", "Staging", "Production", "Archived"]
        else:
            stages = ["None", "Staging", "Production"]
        
        versions = []
        for stage in stages:
            stage_versions = client.get_latest_versions(model_name, stages=[stage])
            versions.extend(stage_versions)
        
        # Sort by version number
        versions.sort(key=lambda v: int(v.version), reverse=True)
        
        print("\n" + "=" * 70)
        print(f"MODEL VERSIONS: {model_name}")
        print("=" * 70)
        print(f"{'Version':<10} {'Stage':<15} {'Status':<12} {'Created':<20}")
        print("-" * 70)
        
        for v in versions:
            created = datetime.fromtimestamp(v.creation_timestamp / 1000).strftime(
                "%Y-%m-%d %H:%M"
            )
            print(f"{v.version:<10} {v.current_stage:<15} {v.status:<12} {created:<20}")
        
        if not versions:
            print("No versions found.")
        
        print()
        return versions
        
    except mlflow.exceptions.MlflowException as e:
        print(f"Error: {e}")
        return []


def cmd_rollback(
    model_name: str,
    target_version: Optional[int] = None,
    tracking_uri: Optional[str] = None,
    archive_current: bool = True,
) -> bool:
    """
    Rollback to a previous model version.
    
    Args:
        model_name: Name of the registered model.
        target_version: Version to rollback to. If None, uses previous production version.
        tracking_uri: MLflow tracking URI.
        archive_current: Archive the current production version.
        
    Returns:
        True if rollback successful.
    """
    client = get_client(tracking_uri)
    
    try:
        # Get current production version
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not prod_versions:
            print(f"No production version found for '{model_name}'")
            return False
        
        current_prod = prod_versions[0]
        current_version = int(current_prod.version)
        
        # Determine target version
        if target_version is None:
            # Default to previous version
            if current_version <= 1:
                print("Cannot rollback: current version is 1 (no previous version)")
                return False
            target_version = current_version - 1
        
        if target_version == current_version:
            print(f"Target version {target_version} is already in production")
            return False
        
        print(f"\nRolling back '{model_name}':")
        print(f"  Current production: v{current_version}")
        print(f"  Target version: v{target_version}")
        
        # Archive current production version
        if archive_current:
            client.transition_model_version_stage(
                name=model_name,
                version=str(current_version),
                stage="Archived",
            )
            print(f"  → Archived v{current_version}")
        
        # Promote target version to production
        client.transition_model_version_stage(
            name=model_name,
            version=str(target_version),
            stage="Production",
            archive_existing_versions=False,
        )
        print(f"  → Promoted v{target_version} to Production")
        
        print("\n✓ Rollback complete!")
        return True
        
    except mlflow.exceptions.MlflowException as e:
        print(f"Error during rollback: {e}")
        return False


def cmd_promote(
    model_name: str,
    version: int,
    stage: str = "Production",
    tracking_uri: Optional[str] = None,
    archive_current: bool = True,
) -> bool:
    """
    Manually promote a model version to a stage.
    
    Args:
        model_name: Name of the registered model.
        version: Version number to promote.
        stage: Target stage (Staging, Production).
        tracking_uri: MLflow tracking URI.
        archive_current: Archive current version in target stage.
        
    Returns:
        True if promotion successful.
    """
    client = get_client(tracking_uri)
    
    if stage not in ["Staging", "Production"]:
        print(f"Invalid stage: {stage}. Must be 'Staging' or 'Production'")
        return False
    
    try:
        # Archive existing version in target stage
        if archive_current:
            existing = client.get_latest_versions(model_name, stages=[stage])
            for v in existing:
                if int(v.version) != version:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived",
                    )
                    print(f"Archived existing {stage} version: v{v.version}")
        
        # Promote target version
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=False,
        )
        
        print(f"\n✓ Promoted '{model_name}' v{version} to {stage}")
        return True
        
    except mlflow.exceptions.MlflowException as e:
        print(f"Error during promotion: {e}")
        return False


def cmd_archive(
    model_name: str,
    version: int,
    tracking_uri: Optional[str] = None,
) -> bool:
    """
    Archive a model version.
    
    Args:
        model_name: Name of the registered model.
        version: Version number to archive.
        tracking_uri: MLflow tracking URI.
        
    Returns:
        True if archive successful.
    """
    client = get_client(tracking_uri)
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Archived",
        )
        
        print(f"✓ Archived '{model_name}' v{version}")
        return True
        
    except mlflow.exceptions.MlflowException as e:
        print(f"Error during archive: {e}")
        return False


def cmd_get_production(
    model_name: str,
    tracking_uri: Optional[str] = None,
) -> Optional[ModelVersion]:
    """
    Get current production model version.
    
    Args:
        model_name: Name of the registered model.
        tracking_uri: MLflow tracking URI.
        
    Returns:
        ModelVersion or None.
    """
    client = get_client(tracking_uri)
    
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if versions:
            v = versions[0]
            print(f"\nProduction model: {model_name}")
            print(f"  Version: {v.version}")
            print(f"  Run ID: {v.run_id}")
            print(f"  Created: {datetime.fromtimestamp(v.creation_timestamp / 1000)}")
            return v
        else:
            print(f"No production version for '{model_name}'")
            return None
            
    except mlflow.exceptions.MlflowException as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# Main CLI
# ============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLflow Model Management CLI",
        prog="python -m mlflow.cli",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list-versions command
    list_parser = subparsers.add_parser(
        "list-versions",
        help="List all versions of a model",
    )
    list_parser.add_argument("--model-name", required=True, help="Model name")
    list_parser.add_argument(
        "--show-archived", action="store_true", help="Include archived versions"
    )
    
    # rollback command
    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Rollback to a previous model version",
    )
    rollback_parser.add_argument("--model-name", required=True, help="Model name")
    rollback_parser.add_argument(
        "--version", type=int, default=None, help="Target version (default: previous)"
    )
    rollback_parser.add_argument(
        "--no-archive", action="store_true", help="Don't archive current version"
    )
    
    # promote command
    promote_parser = subparsers.add_parser(
        "promote",
        help="Manually promote a model version",
    )
    promote_parser.add_argument("--model-name", required=True, help="Model name")
    promote_parser.add_argument("--version", type=int, required=True, help="Version to promote")
    promote_parser.add_argument(
        "--stage",
        choices=["Staging", "Production"],
        default="Production",
        help="Target stage",
    )
    promote_parser.add_argument(
        "--no-archive", action="store_true", help="Don't archive existing version"
    )
    
    # archive command
    archive_parser = subparsers.add_parser(
        "archive",
        help="Archive a model version",
    )
    archive_parser.add_argument("--model-name", required=True, help="Model name")
    archive_parser.add_argument("--version", type=int, required=True, help="Version to archive")
    
    # production command
    prod_parser = subparsers.add_parser(
        "production",
        help="Get current production model",
    )
    prod_parser.add_argument("--model-name", required=True, help="Model name")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "list-versions":
        cmd_list_versions(
            args.model_name,
            args.tracking_uri,
            args.show_archived,
        )
    elif args.command == "rollback":
        success = cmd_rollback(
            args.model_name,
            args.version,
            args.tracking_uri,
            not args.no_archive,
        )
        return 0 if success else 1
    elif args.command == "promote":
        success = cmd_promote(
            args.model_name,
            args.version,
            args.stage,
            args.tracking_uri,
            not args.no_archive,
        )
        return 0 if success else 1
    elif args.command == "archive":
        success = cmd_archive(
            args.model_name,
            args.version,
            args.tracking_uri,
        )
        return 0 if success else 1
    elif args.command == "production":
        cmd_get_production(args.model_name, args.tracking_uri)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
