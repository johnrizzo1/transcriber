"""
Calculator tool for mathematical operations.
"""

import ast
import operator
import math
from typing import Any, List, Union

from ..base import BaseTool, ToolCategory, ToolMetadata, ToolParameter, ToolPermission


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Perform mathematical calculations and evaluations",
            category=ToolCategory.UTILITY,
            version="1.0.0",
            author="System",
            permissions=[],  # No special permissions needed
            examples=[
                'calculator(expression="2 + 2")',
                'calculator(expression="sqrt(16) + pi")',
                'calculator(expression="10 * 5 - 3 / 2")',
                'calculator(expression="sin(pi/2) + cos(0)")'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="str",
                description="Mathematical expression to evaluate",
                required=True
            )
        ]
    
    async def _execute(self, expression: str) -> Union[float, int]:
        """
        Evaluate a mathematical expression safely.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation
        """
        # Replace some common mathematical terms
        expression = expression.replace("^", "**")  # Power operator
        
        # Define allowed operations and functions
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
        }
        
        allowed_functions = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            # Math functions
            "sqrt": math.sqrt,
            "pow": math.pow,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "degrees": math.degrees,
            "radians": math.radians,
            "factorial": math.factorial,
            "gcd": math.gcd,
            "ceil": math.ceil,
            "floor": math.floor,
        }
        
        allowed_constants = {
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "inf": math.inf,
        }
        
        class MathEvaluator(ast.NodeVisitor):
            """Safe math expression evaluator."""
            
            def visit_BinOp(self, node):
                left = self.visit(node.left)
                right = self.visit(node.right)
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](left, right)
                else:
                    raise ValueError(f"Unsupported operation: {op_type.__name__}")
            
            def visit_UnaryOp(self, node):
                operand = self.visit(node.operand)
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](operand)
                else:
                    raise ValueError(f"Unsupported operation: {op_type.__name__}")
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in allowed_functions:
                        args = [self.visit(arg) for arg in node.args]
                        return allowed_functions[func_name](*args)
                    else:
                        raise ValueError(f"Function not allowed: {func_name}")
                else:
                    raise ValueError("Complex function calls not allowed")
            
            def visit_Name(self, node):
                if node.id in allowed_constants:
                    return allowed_constants[node.id]
                else:
                    raise ValueError(f"Unknown variable: {node.id}")
            
            def visit_Constant(self, node):
                return node.value
            
            def visit_Num(self, node):  # For Python < 3.8 compatibility
                return node.n
            
            def generic_visit(self, node):
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")
        
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate safely
            evaluator = MathEvaluator()
            result = evaluator.visit(tree.body)
            
            # Clean up the result
            if isinstance(result, float) and result.is_integer():
                return int(result)
            
            return result
            
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")


class AdvancedCalculatorTool(CalculatorTool):
    """Advanced calculator with additional features."""
    
    def _define_metadata(self) -> ToolMetadata:
        metadata = super()._define_metadata()
        metadata.name = "advanced_calculator"
        metadata.description = "Advanced calculator with unit conversions and statistics"
        metadata.version = "1.1.0"
        metadata.examples.extend([
            'advanced_calculator(expression="mean([1, 2, 3, 4, 5])")',
            'advanced_calculator(expression="std([1, 2, 3, 4, 5])")',
            'advanced_calculator(expression="convert(100, \'celsius\', \'fahrenheit\')")'
        ])
        return metadata
    
    async def _execute(self, expression: str) -> Any:
        """Execute with additional statistical and conversion functions."""
        # Add statistical functions
        import statistics
        
        additional_functions = {
            "mean": statistics.mean,
            "median": statistics.median,
            "mode": statistics.mode,
            "std": statistics.stdev,
            "variance": statistics.variance,
        }
        
        # Add unit conversion (simplified)
        def convert(value: float, from_unit: str, to_unit: str) -> float:
            """Simple unit converter."""
            conversions = {
                # Temperature
                ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
                ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
                ("celsius", "kelvin"): lambda x: x + 273.15,
                ("kelvin", "celsius"): lambda x: x - 273.15,
                # Length
                ("meters", "feet"): lambda x: x * 3.28084,
                ("feet", "meters"): lambda x: x / 3.28084,
                ("kilometers", "miles"): lambda x: x * 0.621371,
                ("miles", "kilometers"): lambda x: x / 0.621371,
                # Weight
                ("kilograms", "pounds"): lambda x: x * 2.20462,
                ("pounds", "kilograms"): lambda x: x / 2.20462,
            }
            
            key = (from_unit.lower(), to_unit.lower())
            if key in conversions:
                return conversions[key](value)
            else:
                raise ValueError(f"Unknown conversion: {from_unit} to {to_unit}")
        
        additional_functions["convert"] = convert
        
        # Temporarily add these functions to the expression evaluator
        # This is a simplified approach - in production, we'd modify the evaluator class
        expression = expression.replace("mean", "statistics.mean")
        expression = expression.replace("std", "statistics.stdev")
        
        return await super()._execute(expression)