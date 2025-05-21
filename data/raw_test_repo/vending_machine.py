from decimal import Decimal
from typing import Optional, List, Tuple
from .models.product import Item
from .payment.payment_processor import Handler, Tx, TxStatus, Cash
from .inventory.inventory_manager import Store

class SysErr(Exception):
    """
    No docstring provided.
    """
    pass

class Sys:
    """
    A comprehensive transaction management system for item selection, purchase, and payment processing.

    Provides a robust, flexible framework for managing inventory transactions with
    integrated payment handling and error management. This system acts as a central
    coordinator between item storage, transaction processing, and payment mechanisms,
    enabling complex purchasing workflows with built-in validation and safety checks.

    The Sys class serves as an orchestrator that combines multiple components:
    - Item storage and management
    - Transaction processing
    - Payment handling
    - Error management

    Key Features:
        - Flexible payment handler support
        - Comprehensive transaction lifecycle management
        - Integrated inventory and transaction validation
        - Supports item selection, purchase, and cancellation
        - Provides detailed error handling

    Design Principles:
        - Decoupled architecture
        - Polymorphic transaction handling
        - Strict validation at each transaction stage
        - Support for different payment methods

    Parameters:
        h (Optional[Handler]): A transaction handler, defaulting to Cash handler
                               if no specific handler is provided.

    Attributes:
        store (Store): Manages item inventory and storage
        h (Handler): Handles transaction processing
        _tx (Optional[Tx]): Tracks the current active transaction

    Use Cases:
        - Vending machine systems
        - Retail point-of-sale applications
        - Self-service kiosks
        - Automated inventory management

    Examples:
        # Initialize system with default or custom handler
        system = Sys()  # Uses default Cash handler
        custom_system = Sys(custom_handler)

        # Typical workflow
        system.add_money(Decimal('10.00'))
        item, change = system.buy(2)
        print(f"Purchased: {item.label}, Change: ${change}")

        # Handle transaction cancellation
        system.cancel()
    """

    def __init__(self, h: Optional[Handler]=None):
        self.store = Store()
        self.h = h or Cash()
        self._tx: Optional[Tx] = None

    def ls(self) -> List[Tuple[int, Item]]:
        """
        No docstring provided.
        """
        items = []
        for item in self.store.ls():
            pos = self.store.find(item.code)
            if pos is not None:
                items.append((pos, item))
        return sorted(items, key=lambda x: x[0])

    def pick(self, pos: int) -> Optional[Item]:
        """
        No docstring provided.
        """
        item = self.store.get_at(pos)
        if not item:
            raise SysErr('invalid pos')
        if not item.check():
            raise SysErr('unavailable')
        return item

    def add_money(self, amt: Decimal) -> None:
        """
        No docstring provided.
        """
        if not isinstance(self.h, Cash):
            raise SysErr('cash not supported')
        self.h.add(amt)

    def buy(self, pos: int) -> Tuple[Item, Optional[Decimal]]:
        """
        No docstring provided.
        """
        item = self.pick(pos)
        tx = self.h.proc(Decimal(str(item.val)))
        self._tx = tx
        if tx.st != TxStatus.DONE:
            raise SysErr(tx.msg or 'tx failed')
        if not item.mod():
            self.h.rev(tx)
            raise SysErr('dispense failed')
        ret = None
        if isinstance(self.h, Cash):
            ret = self.h.ret()
        return (item, ret)

    def cancel(self) -> Optional[Decimal]:
        """
        No docstring provided.
        """
        if not self._tx:
            raise SysErr('no tx')
        ok = self.h.rev(self._tx)
        if not ok:
            raise SysErr('rev failed')
        ret = None
        if isinstance(self.h, Cash):
            ret = self.h.ret()
        self._tx = None
        return ret