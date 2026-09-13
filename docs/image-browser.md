# Image browser

Opening the Image browser tab refreshes the database from the output and archive
folders and returns to page one. Update DB remains available for manual refresh.
The index includes PNG, JPEG, WebP, GIF, MP4 and WebM files. Videos and their GIF
previews remain separate entries, with embedded generation metadata available for
both. Refresh updates database entries; it does not delete output files.

Missing files are removed from the index when loading a page. Displayed images
use Gradio cache copies, so deleting an original while its thumbnail is visible
does not break selection. Metadata selection reports when the original was removed.

Empty searches, missing output folders and refresh failures keep the page slider
valid. Unreadable GIFs and videos are skipped rather than aborting the folder scan.
