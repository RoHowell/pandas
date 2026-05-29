import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestMutate:
    def test_mutate_single_column(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        df.mutate(name="log_x", input_columns=["x"], transform=np.log)
        expected = DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "log_x": [0.0, 0.693147, 1.098612]
        })
        tm.assert_frame_equal(df, expected)

    def test_mutate_multiple_columns(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.mutate(name="sum_a_b", input_columns=["a", "b"], transform=lambda a, b: a + b)
        expected = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "sum_a_b": [5, 7, 9]})
        tm.assert_frame_equal(df, expected)

    def test_mutate_overwrite_existing(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.mutate(name="x", input_columns=["x"], transform=lambda x: x * 2)
        expected = DataFrame({"x": [2, 4, 6], "y": [4, 5, 6]})
        tm.assert_frame_equal(df, expected)

    def test_mutate_missing_column(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        msg = "The following input_columns are missing from the DataFrame: \\['c'\\]"
        with pytest.raises(KeyError, match=msg):
            df.mutate(name="new_col", input_columns=["a", "c"], transform=lambda a, c: a + c)

    def test_mutate_multiple_missing_columns(self):
        df = DataFrame({"a": [1, 2, 3]})
        msg = "The following input_columns are missing from the DataFrame: \\['b', 'c'\\]"
        with pytest.raises(KeyError, match=msg):
            df.mutate(name="new_col", input_columns=["b", "c"], transform=lambda b, c: b + c)

    def test_mutate_in_place(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        original_id = id(df)
        result = df.mutate(name="z", input_columns=["x", "y"], transform=lambda x, y: x + y)
        assert result is None
        assert id(df) == original_id
        assert "z" in df.columns
        tm.assert_series_equal(df["z"], df["x"] + df["y"])

    def test_mutate_with_numpy_function(self):
        df = DataFrame({"a": [1.0, 4.0, 9.0]})
        df.mutate(name="sqrt_a", input_columns=["a"], transform=np.sqrt)
        expected = DataFrame({"a": [1.0, 4.0, 9.0], "sqrt_a": [1.0, 2.0, 3.0]})
        tm.assert_frame_equal(df, expected)

    def test_mutate_three_columns(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        df.mutate(name="sum_abc", input_columns=["a", "b", "c"], transform=lambda a, b, c: a + b + c)
        expected = DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "sum_abc": [12, 15, 18]
        })
        tm.assert_frame_equal(df, expected)
