import nbconvert
import os
import sys
import nbformat
from copy import deepcopy
from nbconvert.preprocessors import TagRemovePreprocessor, ExecutePreprocessor
import traceback
import re 

import asyncio
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        print("Applied WindowsSelectorEventLoopPolicy for asyncio.")
    except AttributeError:
        print("Could not apply WindowsSelectorEventLoopPolicy (requires Python 3.8+).")
    except Exception as e_policy:
         print(f"Warning: Could not set event loop policy: {e_policy}")

# --- Configuration ---
notebook_in_path = 'docs/notebook.ipynb'
output_base_dir = 'docs'
build_tasks = {
    'features/trend_strength.md': ['trend_strength'],
    'features/trend_changes.md': ['trend_changes'],
    'features/linear_regression.md': ['linear_regression'],
    'features/forecastability.md': ['forecastability'],
    'features/fluctuation.md': ['fluctuation'],
    'features/ac_relevance.md': ['ac_relevance'],
    'features/seasonal_strength.md': ['seasonal_strength'],
    'features/window_fluctuation.md': ['window_fluctuation'],
    'features/st_variation.md': ['st_variation'],
    'features/diff_series.md': ['diff_series'],
    'features/complexity.md': ['complexity'],
    'features/rec_concentration.md': ['rec_concentration'],
    'features/centroid.md': ['centroid']
}

# --- Tag Definitions ---
setup_tag = 'setup'
input_removal_tag = 'hide-input'
output_removal_tag = 'hide-output'

# --- Get Absolute Notebook Path and Directory ---
try:
    abs_notebook_path = os.path.abspath(notebook_in_path)
    notebook_dir = os.path.dirname(abs_notebook_path)
    if not os.path.exists(abs_notebook_path):
         raise FileNotFoundError(f"Notebook file not found at calculated path: {abs_notebook_path}")
except Exception as e:
     print(f"ERROR determining notebook path: {e}")
     sys.exit(1)

# --- Preprocessor Setup ---
remover = TagRemovePreprocessor(
    remove_input_tags=[input_removal_tag],
    remove_all_outputs_tags=[output_removal_tag]
)
remover.enabled = True
executor = ExecutePreprocessor(timeout=900, kernel_name='python3')
print(f"Preprocessors initialized (Executor timeout: {executor.timeout}s, Input removal tag: '{input_removal_tag}', Output removal tag: '{output_removal_tag}')")

# --- Conversion Loop ---
success_count = 0
fail_count = 0

for rel_out_path, tags_to_keep in build_tasks.items():
    print(f"\nProcessing for output '{rel_out_path}' with target tags {tags_to_keep}...")

    md_filename_base = os.path.basename(rel_out_path).replace('.md', '')
    resource_prefix = md_filename_base 

    notebook_to_process = None

    try:
        # --- 1. LOAD Notebook  --
        print(f"  Loading notebook: {notebook_in_path}")
        with open(notebook_in_path, 'r', encoding='utf-8') as f:
            original_notebook = nbformat.read(f, as_version=4)
        print(f"  Notebook loaded successfully for this task.")

        # --- 2. FILTERING (Keep Setup + Target Content) ---
        filtered_nb_content = deepcopy(original_notebook)
        filtered_cells = []
        target_cell_found = False
        print(f"  Filtering: Keeping cells tagged with '{setup_tag}' OR {tags_to_keep}")
        for cell_idx, cell in enumerate(original_notebook.cells):
            cell_tags = cell.metadata.get('tags', [])
            is_setup_cell = setup_tag in cell_tags
            is_target_cell = any(tag in cell_tags for tag in tags_to_keep)
            if is_setup_cell or is_target_cell:
                filtered_cells.append(deepcopy(cell))
                if is_target_cell:
                    target_cell_found = True
        if not target_cell_found:
            print(f"  WARNING: No cells found with the target tag '{tags_to_keep}'. Skipping.")
            fail_count += 1
            continue
        filtered_nb_content.cells = filtered_cells
        print(f"  Content filtering kept {len(filtered_nb_content.cells)} relevant cells.")

        # --- 3. EXECUTE the filtered notebook ---
        print(f"  Executing relevant cells...")
        executor.preprocess(filtered_nb_content, {'metadata': {'path': notebook_dir}})
        print(f"  Execution finished successfully.")
        notebook_after_execution = filtered_nb_content

        # --- 4. REMOVE INPUT/OUTPUT based on tags ---
        print(f"  Applying Input/Output removal...")
        notebook_final_for_conversion, _ = remover.preprocess(notebook_after_execution, {})
        print(f"  Input/Output removal finished.")

        # --- 5. CONVERT to Markdown ---
        print(f"  Converting to Markdown...")
        md_exporter = nbconvert.MarkdownExporter()
        (output, resources) = md_exporter.from_notebook_node(notebook_final_for_conversion)
        print(f"  Conversion successful.")

        # --- 6. PROCESS & WRITE resources ---
        output_modified = output 
        if 'outputs' in resources and resources['outputs']:
            print(f"  -> Found {len(resources['outputs'])} output resource(s). Processing...")
            full_out_path = os.path.join(output_base_dir, rel_out_path)
            out_dir = os.path.dirname(full_out_path)
            os.makedirs(out_dir, exist_ok=True) # Ensure dir exists

            for original_filename, file_data in resources['outputs'].items():
                original_basename = os.path.basename(original_filename)

                new_safe_filename = f"{resource_prefix}_{original_basename}" 
                img_write_path = os.path.join(out_dir, new_safe_filename)

                print(f"    Renaming resource '{original_basename}' -> '{new_safe_filename}'")

                try:
                    with open(img_write_path, 'wb') as img_f:
                        img_f.write(file_data)
                except Exception as img_e:
                    print(f"     ! ERROR writing renamed resource {new_safe_filename} to {out_dir}: {img_e}")
                    continue 

                pattern = re.compile(r'(!\[.*?\]\()([^)]*?/)?' + re.escape(original_basename) + r'(\))')
                replacement = r'\1' + new_safe_filename + r'\3' 
                output_modified_new, num_replacements = pattern.subn(replacement, output_modified)

                if num_replacements > 0:
                     print(f"      Replaced {num_replacements} link(s) in Markdown for '{original_basename}' with '{new_safe_filename}'.")
                     output_modified = output_modified_new 
                else:
                     print(f"      WARNING: Could not find link for '{original_basename}' in Markdown output to replace. Image saved but might not be linked correctly.")


        # --- 7. WRITE final modified output ---
        with open(full_out_path, 'w', encoding='utf-8') as f:
            f.write(output_modified) 
        print(f"  -> Successfully wrote {full_out_path}")

        success_count += 1

    except Exception as e:
        print(f"  -> UNHANDLED ERROR during processing for tag(s) '{tags_to_keep}': {e}")
        traceback.print_exc()
        fail_count += 1
    # --- END Task Processing ---

print(f"\nConversion process finished. Success: {success_count}, Failed: {fail_count}")


