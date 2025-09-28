#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structure_detect_n_extract import extract_structure, detect_and_extract
try:
    import jsonschema
    SCHEMA_VALIDATION = True
except ImportError:
    SCHEMA_VALIDATION = False
    print("Warning: jsonschema not available. Install with: pip install jsonschema")

class StructureComparisonTest:
    def __init__(self, markdown_file):
        if not os.path.exists(markdown_file):
            raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
        
        with open(markdown_file, 'r', encoding='utf-8') as f:
            self.test_markdown = f.read()
        
        self.markdown_file = markdown_file
        print(f"Loaded markdown file: {markdown_file}")
    
    def compare_structures(self):
        """Compare outputs from both extractors"""
        print("Testing structure extraction comparison...")
        
        # Extract using markdown_structure_extractor
        print("\n1. Using markdown_structure_extractor:")
        mse_structure = extract_structure(self.test_markdown)
        print(f"   Found {len(mse_structure)} top-level items")
        
        # Extract using detect_and_extract
        print("\n2. Using detect_and_extract:")
        dae_structure = detect_and_extract(self.test_markdown)
        print(f"   Found {len(dae_structure)} top-level items")
        
        # Compare structures
        print("\n3. Comparison Results:")
        self._compare_recursive(mse_structure, dae_structure, "")
        
        # Output to JSON for detailed inspection
        with open('mse_output.json', 'w', encoding='utf-8') as f:
            json.dump(mse_structure, f, indent=2, ensure_ascii=False)
        
        with open('dae_output.json', 'w', encoding='utf-8') as f:
            json.dump(dae_structure, f, indent=2, ensure_ascii=False)
        
        print("\nDetailed outputs saved to mse_output.json and dae_output.json")
        
        # Validate against schema
        if SCHEMA_VALIDATION:
            self._validate_schema(mse_structure, "MSE")
            self._validate_schema(dae_structure, "DAE")
        
        return self._structures_match(mse_structure, dae_structure)
    
    def _compare_recursive(self, mse_items, dae_items, indent):
        """Recursively compare structure items"""
        if len(mse_items) != len(dae_items):
            print(f"{indent}❌ Count mismatch: MSE={len(mse_items)}, DAE={len(dae_items)}")
            return False
        
        for i, (mse_item, dae_item) in enumerate(zip(mse_items, dae_items)):
            # Compare basic fields
            mse_type = mse_item.get('type', '')
            dae_type = dae_item.get('type', '')
            mse_num = mse_item.get('number', '')
            dae_num = dae_item.get('number', '')
            
            if mse_type != dae_type or mse_num != dae_num:
                print(f"{indent}❌ Item {i}: MSE({mse_type} {mse_num}) != DAE({dae_type} {dae_num})")
            else:
                print(f"{indent}✅ Item {i}: {mse_type} {mse_num}")
            
            # Compare nested structures
            for nested_key in ['parts', 'divisions', 'subdivisions', 'sections']:
                mse_nested = mse_item.get(nested_key, [])
                dae_nested = dae_item.get(nested_key, [])
                
                if mse_nested or dae_nested:
                    print(f"{indent}  Checking {nested_key}:")
                    self._compare_recursive(mse_nested, dae_nested, indent + "    ")
    
    def _validate_schema(self, structure, name):
        """Validate structure against JSON schema"""
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'structure_schema.json')
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            jsonschema.validate(structure, schema)
            print(f"✅ {name} structure validates against schema")
        except jsonschema.ValidationError as e:
            print(f"❌ {name} schema validation failed: {e.message}")
        except FileNotFoundError:
            print("❌ Schema file not found: structure_schema.json")
    
    def _structures_match(self, mse_structure, dae_structure):
        """Check if structures match exactly"""
        return json.dumps(mse_structure, sort_keys=True) == json.dumps(dae_structure, sort_keys=True)
    
    def run_test(self):
        """Run the comparison test"""
        print("=" * 60)
        print("STRUCTURE EXTRACTION COMPARISON TEST")
        print("=" * 60)
        
        match = self.compare_structures()
        
        print("\n" + "=" * 60)
        if match:
            print("✅ RESULT: Structures match exactly!")
        else:
            print("❌ RESULT: Structures differ - check outputs for details")
        print("=" * 60)
        
        return match

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_structure_comparison.py <markdown_file>")
        sys.exit(1)
    
    markdown_file = sys.argv[1]
    test = StructureComparisonTest(markdown_file)
    test.run_test()

if __name__ == "__main__":
    main()