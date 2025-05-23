# 代表性题目

## BFS模板
```
首元素入队
while(队列不为空)
{
  s=弹出队头元素。
  print（s）
  s的所有节点入队
}
```

## [1.二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/)
思路：当队列不为空时，队列的元素个数就是当前层的节点数，详见代码。     
代码：
```
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();

        List<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.addLast(root);
        while(!queue.isEmpty())
        {
            List<Integer> tmp = new ArrayList<>();
            int n = queue.size(); //得到当前层的节点数
            for(int i = 0; i < n; i++) //遍历当前层的所有节点
            {
                TreeNode node = queue.removeFirst();
                tmp.add(node.val);
                if(node.left != null) queue.addLast(node.left);
                if(node.right != null) queue.addLast(node.right);
            }
            ans.add(tmp);
        }
        return ans;
    }
}
```

## [2.二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/)
思路：和上一题类似，然后偶数层需要反转。   
代码：
```
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        List<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.addLast(root);
        int depth = 0; //偶数层反转
        while(!queue.isEmpty())
        {
            List<Integer> temp = new ArrayList<>();
            int n = queue.size();
            depth++;
            for(int i = 0; i < n; i++)
            {
                TreeNode node = queue.removeFirst();
                temp.add(node.val);
                if(node.left != null) queue.addLast(node.left);
                if(node.right != null) queue.addLast(node.right);
            }
            if(depth % 2 == 0)
                Collections.reverse(temp);
            ans.add(temp);
        }   
        return ans;              
    }
}
```

## [3.找树左下角的值](https://leetcode.cn/problems/find-bottom-left-tree-value/description/)
思路：反向层序遍历（从右到左），最后一个值就是答案。   
代码：
```
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        List<TreeNode> queue = new LinkedList<>();
        queue.addLast(root);
        int val = 0; //记录反向层序遍历的值，最后一个值是答案
        while(!queue.isEmpty())
        {
            TreeNode node = queue.removeFirst();
            val = node.val;
            if(node.right != null) queue.addLast(node.right);
            if(node.left != null) queue.addLast(node.left);
        }
        return val;
    }
}
```

## [4.二叉树的层序遍历 II](https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/description/)
思路：层序遍历，然后反转最后的答案即可。   
代码：
```
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();

        List<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.addLast(root);
        while(!queue.isEmpty())
        {
            List<Integer> tmp = new ArrayList<>();
            int n = queue.size(); //得到当前层的节点数
            for(int i = 0; i < n; i++) //遍历当前层的所有节点
            {
                TreeNode node = queue.removeFirst();
                tmp.add(node.val);
                if(node.left != null) queue.addLast(node.left);
                if(node.right != null) queue.addLast(node.right);
            }
            ans.add(tmp);
        }
        Collections.reverse(ans);
        return ans;
    }
}
```

## [5.填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/description/)
## [6.填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/description/)
### 方法一：
思路：反向层序遍历，每一层进行next指针的指向即可。   
代码：
```
class Solution {
    public Node connect(Node root) {
        List<Node> queue = new LinkedList<>();

        //反向层序遍历
        if(root != null) queue.addLast(root);
        while(!queue.isEmpty())
        {
            int n = queue.size(); 
            Node nxt = null; //表示下一节点
            for(int i = 0; i < n; i++) 
            {
                Node node = queue.removeFirst();
                node.next = nxt; //当前节点指向下一节点
                nxt = node;  //更新下一节点
                
                if(node.right != null) queue.addLast(node.right);
                if(node.left != null) queue.addLast(node.left);
            }
        }
        return root;
    }
}
```
### 方法二（使用常量级额外空间）：
思路：  
![image](https://github.com/user-attachments/assets/e19b88a6-72ae-4d25-b549-f368b8eaf12a)

代码：
```
class Solution {
    public Node connect(Node root) {
        Node cur = root; //指向当前层的节点
        while(cur != null)
        {
            Node nxtHead = cur.left; //记录下一层的首节点
            Node nxt = new Node(); //下一层的哨兵节点，用于连接下一个节点
            while(cur != null)
            {
                if(cur.left != null)
                {
                    nxt.next = cur.left;
                    nxt = cur.left;
                }
                if(cur.right != null)
                {
                    nxt.next = cur.right;
                    nxt = cur.right;
                }
                cur = cur.next;
            }
            cur = nxtHead; //cur指向下一层的首节点
        }
        return root;
    }
}
```

## [7.反转二叉树的奇数层](https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/description/)
思路：BFS，用ArrayList来代表队列，偶数层时直接把队列复制到一个新的ArrayList并进行交换值操作，队列清空。   
代码：
```
class Solution {
    public TreeNode reverseOddLevels(TreeNode root) {
        List<TreeNode> queue = new ArrayList<>();
        int depth = 0;
        queue.add(root);
        while(!queue.isEmpty())
        {
            List<TreeNode> layer = queue; //拿到该层所有节点
            queue = new ArrayList<>();  //清空队列
            depth++;
            if(depth % 2 == 0) //偶数层，交换值
            {
                int l = 0;
                int r = layer.size()-1;
                while(l <= r)
                {
                    int t = layer.get(l).val;
                    layer.get(l).val = layer.get(r).val;
                    layer.get(r).val = t;
                    l++;
                    r--;
                }
            }
            for(int i = 0; i < layer.size(); i++)
            {
                TreeNode node = layer.get(i);
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
        }
        return root;
    }
}
```

## [8.二叉树的堂兄弟节点 II](https://leetcode.cn/problems/cousins-in-binary-tree-ii/description/)
思路：  
![image](https://github.com/user-attachments/assets/4362305d-5558-4c28-9655-3734655f6d87)
  
代码：
```
class Solution {
    public TreeNode replaceValueInTree(TreeNode root) {
        List<TreeNode> queue = new ArrayList<>();
        root.val = 0;
        queue.add(root);
        while(!queue.isEmpty())
        {
            List<TreeNode> layer = queue; //得到当前层
            queue = new ArrayList<>(); //队列清空
            int sum = 0; //下一层的和
            for(TreeNode node: layer) //第一次循环当前层,计算下一层的和
            {
                if(node.left != null) sum += node.left.val;
                if(node.right != null) sum += node.right.val;
            }
            for(TreeNode node: layer) //第二次循环当前层,更新下一层节点的值
            {
                //计算亲兄弟的和
                int familySum = 0;
                if(node.left != null) familySum += node.left.val;
                if(node.right != null) familySum += node.right.val;
                //更新下一层的值和队列
                if(node.left != null)
                {
                    node.left.val = sum - familySum;
                    queue.add(node.left);
                }
                if(node.right != null)
                {
                    node.right.val = sum - familySum;
                    queue.add(node.right);
                }
            }
        }
        return root;
    }
}
```
