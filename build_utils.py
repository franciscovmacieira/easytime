# File: build_utils.py
from nbconvert.preprocessors import Preprocessor
from traitlets import Unicode, List

class KeepTagPreprocessor(Preprocessor):
    """
    Nbconvert preprocessor to select only cells having a specific tag.
    NOTE: This is NOT used by the current build_docs.py which uses
          manual filtering, but kept here for reference.
    """
    keep_tags = List(Unicode(), [""]).tag(config=True) # Tag(s) to keep

    def preprocess(self, nb, resources):
        # --- Enhanced Debugging ---
        print(f"\n--- Running KeepTagPreprocessor (Instance ID: {id(self)}) ---")
        print(f"    Configured keep_tags: {self.keep_tags}")
        # --- End Enhanced Debugging ---

        if not self.keep_tags or self.keep_tags == [""]:
            # If no tag specified, keep all
            return nb, resources

        initial_cell_count = len(nb.cells) # Store initial count
        filtered_cells = []
        for i, cell in enumerate(nb.cells): # Added index for debugging
            cell_tags = cell.metadata.get('tags', [])
            # Keep cell if ANY of the keep_tags are present in cell_tags
            should_keep = any(tag in cell_tags for tag in self.keep_tags)

            if should_keep:
                filtered_cells.append(cell)

        # --- Enhanced Debugging ---
        print(f"    Initial cell count: {initial_cell_count}")
        print(f"    Filtered cell count: {len(filtered_cells)}")
        if initial_cell_count > 0 and not filtered_cells:
             print(f"    WARNING: No cells kept for tags {self.keep_tags}!")
        print(f"--- Finished KeepTagPreprocessor ---")
        # --- End Enhanced Debugging ---

        nb.cells = filtered_cells
        return nb, resources
