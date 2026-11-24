# %%
from pathlib import Path
import sys, os, json, pprint
try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive window without __file__
    _HERE = Path.cwd()
sys.path += [str(_HERE.parents[1] / 'physion' / 'src'),  # physion source
             str(_HERE.parent)]                          # my_analysis (for utils_)

import physion
from physion.assembling.dataset import read_spreadsheet
import numpy as np
# %%

C_ROOT = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments')
E_ROOT = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments')

DATATABLES = [
    # --- C: (~/DATA/In_Vivo_experiments) ---
    os.path.join(C_ROOT, '2NatIm8contrasts', 'NDNF-Cre', 'DataTable.xlsx'),
    os.path.join(C_ROOT, 'NDNF-old-protocol', 'NDNF-WT-Dec-2022', 'DataTable.xlsx'),
    os.path.join(C_ROOT, 'Ori-contrasts', 'NDNF-Cre', 'DataTable.xlsx'),
    os.path.join(C_ROOT, 'Ori-contrasts', 'SST-cells_WT_Adult_V1', 'DataTable.xlsx'),
    os.path.join(C_ROOT, 'Other', 'NDNF-Cre-batch3', 'DataTable.xlsx'),
    os.path.join(C_ROOT, 'Other', 'SST-cells_WT_Adult_V1', 'DataTable.xlsx'),
    os.path.join(C_ROOT, 'Vision-survey', 'NDNF-Cre', 'DataTable.xlsx'),
    # --- E: (E:/DATA/In_Vivo_experiments) ---
    os.path.join(E_ROOT, '10NatIm2contrasts', 'C57BL6J', 'DataTable.xlsx'),
    os.path.join(E_ROOT, '10NatIm2contrasts', 'NDNF-Cre', 'DataTable.xlsx'),
    os.path.join(E_ROOT, '10NatIm2contrasts', 'Thy1GCaMP6s-NDNF-Cre', 'DataTable.xlsx'),
    os.path.join(E_ROOT, 'opto', 'ffSG-8ori-2contrasts-opto', 'NDNF-Cre', 'DataTable.xlsx'),
    os.path.join(E_ROOT, 'opto', 'ffSG-8ori-2contrasts-opto', 'SST-Cre', 'DataTable.xlsx'),
    os.path.join(E_ROOT, 'opto', 'vision-survey-opto', 'C57BL6J', 'DataTable.xlsx'),
    os.path.join(E_ROOT, 'opto', 'vision-survey-opto', 'NDNF-Cre', 'DataTable.xlsx'),
    os.path.join(E_ROOT, 'opto', 'vision-survey-opto', 'SST-Cre', 'DataTable.xlsx'),
    os.path.join(E_ROOT, 'opto', 'vision-survey-opto', 'Thy1GCaMP-NDNF-Cre', 'DataTable.xlsx'),
]

VERBOSE = False  # True -> print the rotating-disk block of every recording

text_replace = {\
    '"roto-encoder-value-per-rotation": -25300.0':\
        '"roto-encoder-value-per-rotation": -4026.6',
    '"roto-encoder-value-per-rotation": -12563.0':\
        '"roto-encoder-value-per-rotation": -1999.5',
    }
npy_replace = {
    'roto-encoder-value-per-rotation': {-25300.0: -4026.6, -12563.0: -1999.5},
    }


def fix_datatable(DataTable):

    dataset, _, _ = read_spreadsheet(DataTable)
    n_fixed = 0

    for d in dataset['datafolder']:

        if VERBOSE:
            print(' - loading: ', d)

        json_file = os.path.join(d, 'metadata.json')
        npy_file = os.path.join(d, 'metadata.npy')

        if os.path.isfile(json_file):

            with open(json_file, 'r') as f:
                content = f.read()

            if VERBOSE:
                pprint.pprint(json.loads(content).get('rotating-disk'))

            new_content = content
            for key in text_replace:
                new_content = new_content.replace(key, text_replace[key])

            if new_content != content:  # only rewrite files that changed
                with open(json_file, 'w') as f:
                    f.write(new_content)
                print('   [ok] replaced value in: %s ' % json_file)
                n_fixed += 1

        elif os.path.isfile(npy_file):

            metadata = np.load(npy_file, allow_pickle=True).item()

            if VERBOSE:
                pprint.pprint(metadata.get('rotating-disk'))

            # value can be nested in 'rotating-disk' or at the top level
            container = metadata.get('rotating-disk', metadata)
            changed = False
            for key, mapping in npy_replace.items():
                if container.get(key) in mapping:
                    container[key] = mapping[container[key]]
                    changed = True
            if changed:
                np.save(npy_file, metadata)
                print('   [ok] replaced value in: %s ' % npy_file)
                n_fixed += 1

        else:
            print('   [!!] no metadata found in: %s ' % d)

    return len(dataset['datafolder']), n_fixed


summary = []
for DataTable in DATATABLES:
    print('\n=== %s ===' % DataTable)
    if not os.path.isfile(DataTable):
        print('   [!!] DataTable not found, skipped')
        summary.append((DataTable, 0, 0, 'not found'))
        continue
    try:
        n_rec, n_fixed = fix_datatable(DataTable)
        summary.append((DataTable, n_rec, n_fixed, 'ok'))
    except Exception as e:
        print('   [!!] error: %s' % e)
        summary.append((DataTable, 0, 0, 'error: %s' % e))

print('\n\n========== SUMMARY ==========')
for DataTable, n_rec, n_fixed, status in summary:
    print('%3i/%3i fixed  [%s]  %s' % (n_fixed, n_rec, status, DataTable))


# %%
