# Changelog

## [Unreleased] - 2024-12-16

### Added
- **Planner Context Enhancement**: Il Planner ora riceve informazioni sui file esistenti prima di generare piani
  - Nuova funzione `_get_file_summary()`: estrae informazioni chiave dai file (requires, classi, funzioni)
  - Nuova funzione `_get_existing_files_context()`: raccoglie lista file esistenti, test e feature completate
  - Il Planner ora vede cosa è già stato implementato e può mantenere coerenza tra feature

### Improved
- **Executor Context**: L'Executor ora riceve il contesto completo del task originale (primi 1500 caratteri)
- **Planner Instructions**: Istruzioni più dettagliate al Planner per generare `content_instruction` specifiche
- **Coerenza tra file**: Il Planner può ora vedere dipendenze tra file e usare file esistenti invece di crearne di nuovi

### Fixed
- Risolto problema di coerenza: `api.php` ora può vedere che `db.php` esiste e usarlo invece di richiedere `database.php`
- Il Planner mantiene coerenza tra feature successive vedendo cosa è già stato implementato

