"""Dataset replacement without code changes: a new KB and a re-mapped catalogue are added purely through config."""
import shutil

import pandas as pd
import yaml

from config.settings import load_kb_config


def test_new_knowledge_base_via_config_only(workspace, pipeline, employee, public_user):
    cfg_path = workspace / "config/knowledge_bases.yaml"
    original = cfg_path.read_text(encoding="utf-8")
    folder = workspace / "knowledge_bases/it_security"
    folder.mkdir()
    try:
        (folder / "Password_Standard.md").write_text(
            "# IT Password Standard (test)\n\nPasswords must be at least 14 characters long and are rotated every "
            "180 days. Password managers approved by IT Security may be used.\n", encoding="utf-8")
        # catalogue with different column names, mapped through entity_fields only
        pd.DataFrame([{"Device Name": "Falcon Laptop", "Device Code": "DEV-1", "Encryption": "BitLocker full-disk"},
                      {"Device Name": "Heron Tablet", "Device Code": "DEV-2", "Encryption": "Hardware AES-256"}]
                     ).to_excel(folder / "Devices.xlsx", index=False)
        cfg = yaml.safe_load(original)
        cfg["knowledge_bases"]["it_security"] = {
            "display_name": "IT Security Standards", "description": "IT security standards and approved devices",
            "enabled": True, "allowed_roles": ["employee", "admin"], "default_document_roles": ["employee", "admin"],
            "default_category": "IT Security", "source_dirs": ["knowledge_bases/it_security"],
            "entity_fields": ["Device Name", "Device Code"],
            "routing": {"keywords": ["password", "device", "encryption", "laptop"],
                        "examples": ["What is the password policy?", "Which encryption does the laptop use?"]}}
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        load_kb_config.cache_clear()
        from ingestion.pipeline import IngestionPipeline

        results = IngestionPipeline().sync_kb("it_security")
        assert {r.status for r in results} == {"added"}
        from backend.services.chat_service import ChatService

        service = ChatService(llm=None)
        resp = service.answer("How long must passwords be?", employee, persist=False)
        assert resp.knowledge_bases == ["it_security"] and "14 characters" in resp.answer
        device = service.answer("Which encryption does the Falcon Laptop use?", employee, persist=False)
        assert "BitLocker" in device.answer
        assert service.answer("How long must passwords be?", public_user, persist=False).answer_type == "access_denied"
    finally:
        IngestionPipeline().purge_kb("it_security")
        cfg_path.write_text(original, encoding="utf-8")
        load_kb_config.cache_clear()
        shutil.rmtree(folder, ignore_errors=True)
