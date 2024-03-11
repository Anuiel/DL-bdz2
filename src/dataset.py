from pathlib import Path


class Multi228k:
    def __init__(self, path_to_data: Path, dataset_type: str | list[str], source_language: str, target_language: str) -> None:
        
        if isinstance(dataset_type, str):
            dataset_type = [dataset_type]
        elif isinstance(dataset_type, list):
            pass
        else:
            raise ValueError

        self.source_language = []
        self.target_language = []

        for subtype in dataset_type:
            base_path = path_to_data / f'{subtype}.{source_language}-{target_language}.aboba'
            source_path = base_path.with_suffix(f'.{source_language}')
            target_path = base_path.with_suffix(f'.{target_language}')
            if source_path.exists():
                with open(source_path) as f:
                    self.source_language += list(map(lambda s: s.rstrip('\n'), f.readlines()))
            if target_path.exists():
                with open(target_path) as f:
                    self.target_language += list(map(lambda s: s.rstrip('\n'), f.readlines()))

    def __iter__(self):
        return zip(self.source_language, self.target_language)
    
    def __getitem__(self, idx: int):
        return (self.source_language[idx], self.target_language[idx])
    
    def __len__(self):
        return len(self.source_language)
