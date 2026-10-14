import json
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.constraint_audit import parent_accounting
from verifier_dataset.inventory import sha,write


class OfflineDependencyAccounting(unittest.TestCase):
    def test_requires_terminal_snapshot_bound_to_exact_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir()
            write(parent/'RECEIPT.json',dict(job='123'))
            write(parent/'CONSTRAINT_COMPLETE.json',dict(passed=True))
            row=dict(root=str(parent),job='123',accounting='123|COMPLETED|0:0',
                receipt_sha256=sha(parent/'RECEIPT.json'),completion_sha256=sha(parent/'CONSTRAINT_COMPLETE.json'))
            write(root/'DEPENDENCY_ACCOUNTING.json',[row])
            self.assertEqual(parent_accounting(root,parent),'123|COMPLETED|0:0')
            row['accounting']='123|FAILED|1:0';write(root/'DEPENDENCY_ACCOUNTING.json',[row])
            with self.assertRaisesRegex(RuntimeError,'normally'):parent_accounting(root,parent)
            row['accounting']='123|COMPLETED|0:0';write(root/'DEPENDENCY_ACCOUNTING.json',[row])
            write(parent/'CONSTRAINT_COMPLETE.json',dict(passed=False))
            with self.assertRaisesRegex(RuntimeError,'changed'):parent_accounting(root,parent)


if __name__=='__main__':unittest.main()
