from typing import List


class DomainParser:
    """Utility class for parsing domain suggestions from model output"""

    @staticmethod
    def parse_domain_suggestions(content: str) -> List[str]:
        """Parse domain suggestions from model output"""
        lines = content.split('\n')
        domains = []

        for line in lines:
            line = line.strip()
            # Look for lines that might contain domains
            if '.' in line and len(line) > 3:
                # Extract potential domains (basic cleaning)
                words = line.split()
                for word in words:
                    if '.' in word and len(word) > 3:
                        # Basic domain validation
                        if word.count('.') >= 1 and not word.startswith('.') and not word.endswith('.'):
                            domains.append(word.lower())

        # If no domains found in expected format, return the raw content split by common separators
        if not domains:
            # Fallback: split content and look for domain-like strings
            content_clean = content.replace(',', '\n').replace(';', '\n')
            domains = [line.strip() for line in content_clean.split('\n') if line.strip()]

        return domains
