"""Command-line interface for ingestion pipeline."""

from pathlib import Path

import click

from common import configure_logging, get_settings


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """Document ingestion pipeline CLI."""
    settings = get_settings()
    configure_logging(
        level="DEBUG" if debug else settings.log_level,
        format=settings.log_format,
    )


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--start-page", "-s", type=int, help="First page to process (1-indexed)")
@click.option("--end-page", "-e", type=int, help="Last page to process (1-indexed)")
@click.option(
    "--strategy",
    type=click.Choice(["semantic", "contextual"]),
    default=None,
    help="Chunking strategy: semantic (fast) or contextual (better retrieval)",
)
def ingest(
    pdf_path: Path,
    start_page: int | None,
    end_page: int | None,
    strategy: str | None,
):
    """Ingest a PDF document into the vector store.

    PDF_PATH: Path to the PDF file to ingest.

    Examples:
        ingestion ingest document.pdf
        ingestion ingest document.pdf --start-page 1 --end-page 10
        ingestion ingest document.pdf --strategy semantic
    """
    from ingestion.pipeline import IngestionPipeline

    click.echo(f"Ingesting: {pdf_path}")

    if start_page or end_page:
        click.echo(f"Page range: {start_page or 1} to {end_page or 'end'}")

    if strategy:
        click.echo(f"Chunking strategy: {strategy}")

    # Progress callback for CLI
    def progress(stage: str, current: int, total: int):
        if total > 0:
            pct = (current / total) * 100
            click.echo(f"  {stage}: {current}/{total} ({pct:.0f}%)")

    # Initialize and run pipeline
    pipeline = IngestionPipeline(
        chunking_strategy=strategy,
    )

    result = pipeline.ingest(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        progress_callback=progress,
    )

    if result["success"]:
        click.echo("\nIngestion complete!")
        click.echo(f"  Document ID: {result['document_id']}")
        click.echo(f"  Pages processed: {result['pages_processed']}")
        click.echo(f"  Chunks created: {result['chunks_created']}")
    else:
        click.echo(f"\nIngestion failed: {result['error']}", err=True)
        raise SystemExit(1)


@cli.command("list")
def list_documents():
    """List all ingested documents."""
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    documents = pipeline.list_documents()

    if not documents:
        click.echo("No documents ingested yet.")
        return

    click.echo(f"Ingested documents ({len(documents)}):\n")

    for doc in documents:
        click.echo(f"  ID: {doc['document_id']}")
        click.echo(f"  Name: {doc['document_name']}")
        click.echo(f"  Chunks: {doc['chunk_count']}")
        click.echo(f"  Pages: {doc['page_count']}")
        click.echo()


@cli.command()
@click.argument("document_id")
def delete(document_id: str):
    """Delete a document from the vector store.

    DOCUMENT_ID: The document ID to delete (from 'list' command).
    """
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    deleted = pipeline.delete_document(document_id)

    if deleted:
        click.echo(f"Deleted {deleted} chunks for document {document_id}")
    else:
        click.echo(f"No chunks found for document {document_id}")


@cli.command()
@click.argument("query")
@click.option("-n", "--num-results", default=5, help="Number of results")
@click.option("-d", "--document-id", help="Filter by document ID")
def search(query: str, num_results: int, document_id: str | None):
    """Search the vector store.

    QUERY: The search query text.

    Examples:
        ingestion search "What is the capital of France?"
        ingestion search "machine learning" -n 10
    """
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    results = pipeline.query(
        query_text=query,
        n_results=num_results,
        document_id=document_id,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        click.echo(f"--- Result {i} ---")
        click.echo(f"Document: {metadata.get('document_name', 'Unknown')}")
        click.echo(f"Page: {metadata.get('page_number', 'N/A')}")
        click.echo(f"Distance: {result.get('distance', 'N/A'):.4f}")
        click.echo(f"Text preview: {result['text'][:200]}...")
        click.echo()


if __name__ == "__main__":
    cli()
