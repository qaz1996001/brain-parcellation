CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `erp_tool_grp_bt`
--

DROP TABLE IF EXISTS `erp_tool_grp_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `erp_tool_grp_bt` (
  `cate1` varchar(16) NOT NULL DEFAULT '*' COMMENT 'different ws_type with differetnt definition.\nex: forming to be mold_id\n-- 2019/12/08 lkchena',
  `ws_type` varchar(12) NOT NULL DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `ws_func` varchar(24) DEFAULT NULL,
  `tool_grp_id` varchar(24) NOT NULL,
  `tool_grp` varchar(32) DEFAULT NULL COMMENT 'tool_group for mold dispatch, \nex: 成型機: 100T成形,150T成形,20T成形,250T成形...\n2019/12/07 lkchena\n',
  `value1` double DEFAULT NULL COMMENT 'reseve for dispatch: ex: forming tons can down grade to run 2020/01/24 lkchena',
  `value2` double DEFAULT NULL,
  `value3` double DEFAULT NULL,
  `param1` varchar(24) DEFAULT NULL COMMENT 'reseve for dispatch: ex: forming tons can down grade to run 2020/01/24 lkchena',
  `param2` varchar(24) DEFAULT NULL,
  `param3` varchar(24) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`cate1`,`ws_type`,`tool_grp_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `erp_tool_grp_bt`
--

LOCK TABLES `erp_tool_grp_bt` WRITE;
/*!40000 ALTER TABLE `erp_tool_grp_bt` DISABLE KEYS */;
INSERT INTO `erp_tool_grp_bt` VALUES ('*','10','成形','FORMING-100T','100T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876'),('*','10','成形','FORMING-150T','150T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876'),('*','10','成形','FORMING-20T','20T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876'),('*','10','成形','FORMING-250T','250T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876'),('*','10','成形','FORMING-350T','350T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876'),('*','10','成形','FORMING-50T','50T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876'),('*','10','成形','FORMING-60T','60T成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-02-16 01:13:51.441876');
/*!40000 ALTER TABLE `erp_tool_grp_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:36
