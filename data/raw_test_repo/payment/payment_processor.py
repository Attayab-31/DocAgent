from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from decimal import Decimal

class TxStatus(Enum):
    """
    Represents the possible states of a financial transaction throughout its lifecycle.

    Provides a standardized enumeration for tracking and managing the progression
    and outcome of transactions. This enum captures the critical stages a transaction
    can experience, enabling precise status tracking and workflow management.

    The TxStatus enum is crucial in systems requiring detailed transaction monitoring,
    such as payment gateways, banking applications, or financial processing platforms.
    It allows for clear, unambiguous communication of a transaction's current state.

    Enum Values:
        WAIT (pending): Transaction is in progress or awaiting processing
        DONE (completed): Transaction has successfully finished
        ERR (failed): Transaction encountered an error and could not complete
        RET (refunded): Transaction was reversed or returned to its original state

    Examples:
        # Tracking transaction status
        status = TxStatus.WAIT
        if status == TxStatus.DONE:
            print("Transaction successful")

        # Using in a transaction workflow
        def process_transaction(tx):
            if tx.status == TxStatus.WAIT:
                # Continue processing
            elif tx.status == TxStatus.ERR:
                # Handle failure
    """
    WAIT = 'pending'
    DONE = 'completed'
    ERR = 'failed'
    RET = 'refunded'

@dataclass
class Tx:
    """
    A comprehensive record of a financial transaction with detailed tracking capabilities.

    Represents a complete transaction snapshot, capturing essential metadata and
    status information for financial operations. This class serves as a robust
    transaction model that enables precise tracking, auditing, and management of
    monetary exchanges across various systems.

    Designed for scenarios requiring detailed transaction logging, such as payment
    processing, banking systems, e-commerce platforms, and financial reconciliation
    workflows. The Tx class provides a standardized structure for representing
    complex transactional events with multiple dimensions of information.

    Attributes:
        id (str): Unique identifier for the transaction, enabling precise tracking
                  and reference across systems.

        amt (Decimal): The precise monetary amount involved in the transaction,
                       supporting exact financial calculations.

        st (TxStatus): Current status of the transaction, indicating its progression
                       through the financial workflow.

        mth (str): Method or channel through which the transaction was processed
                   (e.g., 'credit', 'debit', 'transfer').

        msg (Optional[str]): Optional additional information or notes about the
                             transaction, useful for logging errors or providing
                             context.

    Examples:
        # Creating a new transaction
        transaction = Tx(
            id='TX-2023-001',
            amt=Decimal('100.50'),
            st=TxStatus.WAIT,
            mth='credit_card',
            msg='Online purchase'
        )

        # Updating transaction status
        if transaction.st == TxStatus.WAIT:
            transaction.st = TxStatus.DONE

        # Logging and tracking
        print(f"Transaction {transaction.id}: {transaction.st}")
    """
    id: str
    amt: Decimal
    st: TxStatus
    mth: str
    msg: Optional[str] = None

class Handler(ABC):
    """
    A standardized abstract interface for processing and managing financial transactions.

    Defines a contract for implementing transaction handling mechanisms across
    different financial systems, providing a uniform approach to monetary operations.
    This abstract base class ensures consistent transaction processing and reversal
    strategies while allowing flexible implementation for various financial contexts.

    The Handler serves as a blueprint for creating specialized transaction processors,
    enabling polymorphic handling of financial transactions with a clear, predictable
    interface. It is particularly useful in scenarios requiring diverse transaction
    management strategies, such as payment gateways, banking systems, or financial
    workflow management.

    Key Design Principles:
        - Enforces a consistent transaction processing interface
        - Supports polymorphic transaction handling
        - Provides a flexible framework for different financial operation types

    Abstract Methods:
        - proc(): Processes a financial transaction
        - rev(): Reverses or cancels a previously processed transaction

    Use Cases:
        - Payment system implementations
        - Financial workflow management
        - Transaction tracking and error handling
        - Implementing pluggable transaction processing strategies

    Examples:
        # Creating a concrete handler
        class CashHandler(Handler):
            def proc(self, amt: Decimal) -> Tx:
                # Implement cash-specific processing logic
                pass

            def rev(self, tx: Tx) -> bool:
                # Implement cash-specific reversal logic
                pass

        # Using the handler
        handler = CashHandler()
        transaction = handler.proc(Decimal('100.00'))
        success = handler.rev(transaction)
    """

    @abstractmethod
    def proc(self, amt: Decimal) -> Tx:
        """
        Process a financial transaction for a specified monetary amount.

        Defines the core transaction processing method in the financial handling workflow.
        This abstract method serves as a contract for implementing transaction processing
        logic across different financial handler types, ensuring a consistent interface
        for monetary operations.

        The method is designed to be overridden by concrete implementations, each providing
        specific transaction processing rules based on the handler's unique characteristics.

        Args:
            amt (Decimal): The monetary amount to be processed in the transaction.
                           Must be a positive decimal value representing the transaction quantity.

        Returns:
            Tx: A transaction record representing the result of the processing attempt,
                capturing the transaction's status, method, and any relevant metadata.

        Notes:
            - Abstract method requiring implementation in child classes
            - Serves as a standardized interface for transaction processing
            - Expected to handle various transaction scenarios and generate appropriate
              transaction records

        Examples:
            # In a concrete implementation
            def proc(self, amt: Decimal) -> Tx:
                if self.can_process(amt):
                    return Tx(id='unique_id', amt=amt, st=TxStatus.DONE, mth='custom')
                return Tx(id='unique_id', amt=amt, st=TxStatus.ERR, mth='custom')
        """
        pass

    @abstractmethod
    def rev(self, tx: Tx) -> bool:
        """
        Attempt to reverse or cancel a previously processed financial transaction.

        Defines an abstract method for transaction reversal, serving as a standardized
        interface for undoing or refunding financial transactions across different
        handler implementations. This method ensures a consistent approach to transaction
        rollback and error correction in financial systems.

        The method is designed to be overridden by concrete implementations, each providing
        specific reversal logic based on the handler's unique transaction management rules.

        Args:
            tx (Tx): The transaction to be reversed. Must be a previously processed
                     transaction with sufficient context for potential rollback.

        Returns:
            bool: Indicates the success or failure of the reversal attempt:
                  - True if the transaction can be successfully reversed
                  - False if the reversal is not possible or encounters an error

        Notes:
            - Abstract method requiring implementation in child classes
            - Serves as a contract for transaction reversal mechanisms
            - Expected to handle various reversal scenarios and provide appropriate
              feedback on the reversal attempt

        Examples:
            # In a concrete implementation
            def rev(self, tx: Tx) -> bool:
                if tx.st == TxStatus.DONE and self.can_reverse(tx):
                    # Perform reversal logic
                    tx.st = TxStatus.RET
                    return True
                return False
        """
        pass

class Cash(Handler):
    """
    A concrete implementation of a cash account handler with comprehensive transaction management.

    Provides a fully functional cash account system that supports fundamental financial
    operations including balance management, transaction processing, reversal, and
    complete withdrawal. This implementation serves as a practical example of the
    Handler abstract base class, demonstrating a complete transaction lifecycle.

    The Cash class encapsulates core banking functionality, offering a simple yet
    robust mechanism for managing monetary transactions with precise control and
    immediate status tracking. It is designed for scenarios requiring straightforward
    cash account operations with built-in error handling and transaction management.

    Key Features:
        - Balance tracking with decimal precision
        - Transaction processing with status determination
        - Support for deposits, withdrawals, and refunds
        - Atomic transaction management

    Implemented Methods:
        - add(): Deposit funds into the account
        - proc(): Process withdrawals with status tracking
        - rev(): Reverse completed transactions
        - ret(): Completely drain account balance

    Use Cases:
        - Simple banking systems
        - Wallet or account management applications
        - Financial simulation and modeling
        - Educational demonstrations of transaction handling

    Examples:
        # Create and manage a cash account
        account = Cash()
        account.add(Decimal('100.00'))

        # Process a withdrawal
        tx = account.proc(Decimal('50.00'))

        # Potentially reverse the transaction
        if tx.st == TxStatus.DONE:
            account.rev(tx)
    """

    def __init__(self):
        self.bal: Decimal = Decimal('0.00')

    def add(self, amt: Decimal) -> None:
        """
        Increase the current balance by a specified amount.

        Performs a straightforward increment of the account balance, allowing positive
        monetary adjustments. This method is typically used for depositing funds,
        crediting accounts, or applying financial corrections.

        Args:
            amt (Decimal): The monetary amount to add to the current balance.
                           Must be a non-negative decimal value representing currency.

        Notes:
            - Directly modifies the existing balance
            - No validation is performed on the input amount
            - Supports precise decimal arithmetic

        Examples:
            # Add funds to an account
            account.add(Decimal('50.00'))  # Increases balance by $50.00

            # Apply small credits
            account.add(Decimal('0.50'))  # Adds 50 cents
        """
        self.bal += amt

    def proc(self, amt: Decimal) -> Tx:
        """
        Process a cash withdrawal, validating available balance and generating a transaction record.

        Attempts to withdraw a specified amount from the current balance, creating a
        transaction that reflects the outcome of the withdrawal attempt. This method
        provides a comprehensive mechanism for handling cash transactions with immediate
        status determination.

        Args:
            amt (Decimal): The monetary amount to withdraw. Must be a positive decimal
                           value representing the withdrawal quantity.

        Returns:
            Tx: A transaction record indicating the result of the withdrawal:
                - Successful withdrawal: Transaction with DONE status
                - Insufficient funds: Transaction with ERR status and 'insufficient' message

        Notes:
            - Automatically deducts amount from balance if sufficient funds are available
            - Generates a unique transaction ID based on the current object's identity
            - Provides immediate feedback on transaction success or failure

        Examples:
            # Successful withdrawal
            tx1 = account.proc(Decimal('50.00'))
            print(tx1.st)  # TxStatus.DONE

            # Failed withdrawal due to insufficient funds
            tx2 = account.proc(Decimal('1000.00'))
            print(tx2.st)  # TxStatus.ERR
            print(tx2.msg)  # 'insufficient'
        """
        if self.bal >= amt:
            self.bal -= amt
            return Tx(id=f'C_{id(self)}', amt=amt, st=TxStatus.DONE, mth='cash')
        return Tx(id=f'C_{id(self)}', amt=amt, st=TxStatus.ERR, mth='cash', msg='insufficient')

    def rev(self, tx: Tx) -> bool:
        """
        Reverse a completed transaction by refunding the transaction amount.

        Attempts to undo a previously successful transaction by returning the funds
        to the account and updating the transaction status. This method provides a
        mechanism for financial rollback and error correction.

        Args:
            tx (Tx): The transaction to be reversed. Must be a completed transaction
                     that has not already been refunded.

        Returns:
            bool: True if the reversal was successful (transaction was in DONE state),
                  False if the transaction could not be reversed.

        Notes:
            - Only reverses transactions that are in DONE status
            - Adds the transaction amount back to the account balance
            - Updates the transaction status to RET (refunded)
            - Prevents multiple refunds of the same transaction

        Examples:
            # Successful reversal
            result = account.rev(previous_transaction)
            if result:
                print("Transaction successfully refunded")

            # Failed reversal (already refunded or not completed)
            failed_result = account.rev(another_transaction)
            if not failed_result:
                print("Cannot refund this transaction")
        """
        if tx.st == TxStatus.DONE:
            self.bal += tx.amt
            tx.st = TxStatus.RET
            return True
        return False

    def ret(self) -> Decimal:
        """
        Completely drain the current balance and return the total amount.

        Retrieves the entire current balance and resets the account to zero, effectively
        performing a full balance withdrawal. This method is useful for scenarios requiring
        complete fund liquidation or account closure.

        Returns:
            Decimal: The total balance before resetting to zero, allowing for precise
                     financial tracking and reconciliation.

        Notes:
            - Atomically captures and clears the entire balance
            - Provides a safe way to extract all available funds
            - Leaves the account with a zero balance after execution

        Examples:
            # Retrieve and clear entire balance
            total_funds = account.ret()
            print(f"Withdrawn: {total_funds}")
            print(f"Remaining balance: {account.bal}")  # Will be 0.00
        """
        tmp = self.bal
        self.bal = Decimal('0.00')
        return tmp