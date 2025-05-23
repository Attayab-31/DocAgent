#include <iostream>

/**
 * ```
 * Calculates the sum of two integers.
 * 
 * Summary:
 * This function calculates the sum of two integers.
 * 
 * Description:
 * The calculate_sum function takes two integers as arguments and returns their sum. This function is useful for performing simple arithmetic operations. The function does not modify any other variables or perform any other operations.
 * 
 * Args:
 * a (int) - The first integer to be added.
 * b (int) - The second integer to be added.
 * 
 * Returns:
 * int - The sum of the two integers.
 * 
 * Examples:
 * int result = calculate_sum(3, 5); // result will be 8
 * int result2 = calculate_sum(-2, 7); // result2 will be 5
 * ```
 */
int calculate_sum(int a, int b) {
    return a + b;
}

/**
 * ```
 * A simple calculator for performing basic arithmetic operations.
 * 
 * Summary:
 * This class provides a simple calculator for performing basic arithmetic operations such as addition and subtraction.
 * 
 * Description:
 * The Calculator class is designed to help users perform basic arithmetic operations in a simple and efficient manner. It is useful in scenarios where complex mathematical operations are not required. The class uses private data members to store the current value and provides public methods for addition and subtraction.
 * 
 * Example:
 * Calculator calculator;
 * calculator.add(5);
 * calculator.subtract(3);
 * int result = calculator.get_value(); // result will be 2
 * 
 * Calculator calculator2(10);
 * calculator2.add(2);
 * int result2 = calculator2.get_value(); // result2 will be 12
 * 
 * Parameters:
 * Calculator() - Initializes the calculator with a default value of 0.
 * 
 * Attributes:
 * value - The current value of the calculator. Its type is int and it can hold any integer value.
 * ```
 */
class Calculator {
private:
    int value;

public:
    Calculator() : value(0) {}

/**
     * ```
     * Adds a given integer to the current value.
     * 
     * Summary:
     * This method adds a given integer to the current value.
     * 
     * Description:
     * The add method takes an integer as an argument and adds it to the current value. This method is useful for incrementing the value by a specific amount. The method does not return any value, but instead modifies the current value directly.
     * 
     * Args:
     * x (int) - The integer to be added to the current value.
     * 
     * Examples:
     * Calculator calculator;
     * calculator.add(5);
     * calculator.get_value() // returns 5
     * 
     * calculator.add(3);
     * calculator.get_value() // returns 8
     * ```
     */
    void add(int x) {
        value += x;
    }

   
/**
     * ```
     * Subtracts a given integer from the current value.
     * 
     * Summary:
     * This method subtracts a given integer from the current value.
     * 
     * Description:
     * The subtract method takes an integer as an argument and subtracts it from the current value. This method is useful for decrementing the value by a specific amount. The method does not return any value, but instead modifies the current value directly.
     * 
     * Args:
     * x (int) - The integer to be subtracted from the current value.
     * 
     * Examples:
     * Calculator calculator;
     * calculator.add(5);
     * calculator.subtract(3);
     * calculator.get_value() // returns 2
     * 
     * calculator.subtract(2);
     * calculator.get_value() // returns 0
     * ```
     * 
     * The value variable in the given code component is a private data member of the Calculator class, which stores the current value. It is used by the subtract method to subtract the given integer from the current value.
     */
    void subtract(int x) {
        value -= x;
    }
   
/**
     * ```
     * Retrieves the current value.
     * 
     * Summary:
     * This method retrieves the current value.
     * 
     * Description:
     * The get_value method does not take any arguments and returns the current value. This method is useful for retrieving the current value without modifying it. The method does not modify any other variables or perform any other operations.
     * 
     * Returns:
     * int - The current value.
     * 
     * Examples:
     * Calculator calculator;
     * calculator.add(5);
     * int value = calculator.get_value(); // value will be 5
     * 
     * calculator.subtract(3);
     * int value2 = calculator.get_value(); // value2 will be 2
     * ```
     * 
     * The value variable in the given code component is a private data member of the Calculator class, which stores the current value. It is used by the get\_value method to return the current value.
     */
    int get_value() const {
        return value;
    }
};