from typing import Dict, List, Optional
from ..models.product import Item

class Store:
    """
    A flexible, position-aware inventory management system with fixed-capacity storage.

    Provides a robust mechanism for storing, tracking, and managing items with precise
    control over their placement and accessibility. This class acts as a specialized
    container that supports item insertion, retrieval, removal, and location tracking
    with built-in capacity constraints.

    The Store class is designed for scenarios requiring structured item management,
    such as inventory systems, resource pools, or game item repositories. It offers
    a unique combination of positional and code-based item access, enabling complex
    inventory management workflows.

    Key features include:
    - Fixed-capacity storage with configurable size
    - Dual-mapping system (by code and position)
    - Flexible item placement strategies
    - Efficient item lookup and management

    Parameters:
        cap (int, optional): Maximum number of storage positions. Defaults to 20.

    Attributes:
        cap (int): Total storage capacity of the store
        _data (Dict[str, Item]): Internal storage mapping items by their unique code
        _map (Dict[int, str]): Mapping of storage positions to item codes

    Examples:
        # Create a store with default or custom capacity
        store = Store()  # 20-slot store
        large_store = Store(cap=100)  # 100-slot store

        # Add items with flexible placement
        store.put(item1)  # Auto-place
        store.put(item2, pos=5)  # Specific placement

        # Retrieve and manage items
        item = store.get('ITEM123')
        position = store.find('ITEM123')
        active_items = store.ls()
    """

    def __init__(self, cap: int=20):
        self.cap = cap
        self._data: Dict[str, Item] = {}
        self._map: Dict[int, str] = {}

    def put(self, obj: Item, pos: Optional[int]=None) -> bool:
        """
        Add an item to the store, either at a specific position or in the first available slot.

        Manages item insertion into the store with flexible placement options. If the item
        already exists, it increments the existing item's count. Otherwise, it attempts to
        place the item in a specified or automatically selected storage position.

        Args:
            obj (Item): The item to be added to the store.
            pos (Optional[int], optional): A specific storage position for the item.
                                           If None, an automatic placement is attempted.

        Returns:
            bool: True if the item was successfully added or merged, False if:
                  - The specified position is out of bounds
                  - The specified position is already occupied
                  - No empty positions are available

        Notes:
            - Supports adding new items or incrementing count of existing items
            - Automatically finds the first available slot if no position is specified
            - Prevents overwriting existing items at a given position

        Examples:
            # Add an item to a specific position
            success = store.put(new_item, pos=5)

            # Add an item to the first available slot
            success = store.put(another_item)

            # Handle placement failures
            if not success:
                print("Could not add item to store")
        """
        if obj.code in self._data:
            curr = self._data[obj.code]
            curr.count += obj.count
            return True
        if pos is not None:
            if pos < 0 or pos >= self.cap:
                return False
            if pos in self._map:
                return False
            self._map[pos] = obj.code
        else:
            for i in range(self.cap):
                if i not in self._map:
                    self._map[i] = obj.code
                    break
            else:
                return False
        self._data[obj.code] = obj
        return True

    def rm(self, code: str) -> bool:
        """
        Remove an item from the store by its unique code.

        Completely eliminates an item from both the data storage and location mapping,
        ensuring that all references to the item are thoroughly deleted. This method
        provides a clean way to remove items from the inventory system.

        Args:
            code (str): The unique identifier of the item to be removed.

        Returns:
            bool: True if the item was successfully removed, False if the item
                  does not exist in the store.

        Notes:
            - Removes the item from both internal data storage and location mapping
            - Performs a complete cleanup of item references
            - Silently handles attempts to remove non-existent items

        Examples:
            # Remove an existing item
            removed = store.rm('ITEM123')
            if removed:
                print("Item successfully removed")

            # Attempt to remove a non-existent item
            result = store.rm('NONEXISTENT')
            if not result:
                print("Item not found, no removal performed")
        """
        if code not in self._data:
            return False
        for k, v in list(self._map.items()):
            if v == code:
                del self._map[k]
        del self._data[code]
        return True

    def get(self, code: str) -> Optional[Item]:
        """
        Retrieve an item from the store by its unique code.

        Fetches a specific item from the store's internal data storage using
        the provided item code. This method allows direct access to stored
        items without modifying the store's contents.

        Args:
            code (str): The unique identifier of the item to retrieve.

        Returns:
            Optional[Item]: The item corresponding to the given code, or None
                            if no item with the specified code exists in the store.

        Notes:
            - This is a read-only operation that does not modify the store
            - Returns None if the item code is not found
            - Useful for checking item details without removing the item

        Examples:
            # Retrieve an existing item
            item = store.get('ITEM123')
            if item:
                print(f"Item found: {item.name}, Quantity: {item.count}")

            # Handling non-existent items
            missing_item = store.get('NONEXISTENT')
            if missing_item is None:
                print("Item not found in store")
        """
        return self._data.get(code)

    def get_at(self, pos: int) -> Optional[Item]:
        """
        Retrieve an item from a specific storage position in the store.

        Allows direct access to an item by its physical location within the store's
        storage capacity. This method is useful for inventory management and
        location-based item retrieval, providing a way to access items by their
        assigned storage index.

        Args:
            pos (int): The storage position (index) to retrieve an item from.
                       Must be within the store's capacity range.

        Returns:
            Optional[Item]: The item located at the specified position, or None
                            if no item exists at that storage index.

        Notes:
            - Returns None if the specified position is empty
            - Useful for inventory systems with fixed storage locations
            - Does not modify the store's contents

        Examples:
            # Retrieve an item from a specific storage position
            item = store.get_at(5)
            if item:
                print(f"Item at position 5: {item.name}")
            else:
                print("No item at position 5")

            # Check availability of a storage position
            if store.get_at(10) is None:
                print("Storage position 10 is empty")
        """
        if pos not in self._map:
            return None
        code = self._map[pos]
        return self._data.get(code)

    def ls(self) -> List[Item]:
        """
        List all valid items currently stored in the inventory.

        Retrieves and returns a collection of active items from the store's data,
        filtering out expired or depleted items. This method provides a quick way
        to get a snapshot of currently usable inventory items.

        Returns:
            List[Item]: A list of items that are currently valid (non-expired and
                        with a count greater than zero), representing the active
                        inventory.

        Notes:
            - Uses the Item.check() method to validate each item
            - Filters out items with zero count or past expiration
            - Useful for inventory management and tracking available resources

        Examples:
            # Get all currently valid items
            active_items = store.ls()
            for item in active_items:
                print(f"Active Item: {item.code}, Quantity: {item.count}")

            # Check inventory status
            if not active_items:
                print("No active items in inventory")
        """
        return [obj for obj in self._data.values() if obj.check()]

    def find(self, code: str) -> Optional[int]:
        """
        Locate the storage position of an item by its unique code.

        Searches through the store's internal mapping to find the specific index
        where an item with the given code is stored. This method is useful for
        determining the exact location of an item within the store's capacity.

        Args:
            code (str): The unique identifier of the item to locate.

        Returns:
            Optional[int]: The storage index of the item if found, or None if
                           the item does not exist in the store.

        Examples:
            # Assuming a store with items already placed
            position = store.find('ITEM123')  # Returns the index of 'ITEM123'
            if position is not None:
                print(f"Item found at position {position}")
            else:
                print("Item not found in store")
        """
        for k, v in self._map.items():
            if v == code:
                return k
        return None